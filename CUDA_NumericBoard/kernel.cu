/*
    C++/CUDA Accelerator for the NumericBoard Problem
    --------------------------------------------------------
    This solution leverages the GPU's massive parallelism by having
    the CPU distribute a wide range of subproblems to the CUDA cores.
    Specifically, it utilizes an RTX 3090 FE with 10496 CUDA cores.
    The implementation also features enhanced pruning techniques on both
    the CPU and GPU, significantly reducing the computation time.
    In some cases, this leads to a speedup of up to 1000x due to GPU
    parallelization and other optimizations.

    º The NumericBoard problem, while inherently sequential, is tackled
    through a parallelization strategy. The CPU explores the first N levels
    of the search tree, generating a list of partially solved states,
    which become individual subproblems. These subproblems are then
    dispatched to the GPU. Each GPU thread, assigned a subproblem,
    executes an iterative backtracking algorithm. Upon finding a solution,
    the global atomic counter is incremented to ensure thread-safe accumulation.

    !) A crucial parameter is `N_BRANCHING_LEVELS`, defined within the
    `generate_subproblems` function. Typical values for these branching
    levels range from [4-6]; for particularly large problems, values
    between [7-9] may be more effective.
    --------------------------------------------------------
    [Author: Ignacio Fernández Suárez]
*/

#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <cuda_runtime.h>
#include <cmath>

#define MAX_REAL_SIZE 8
#define MAX_BOARD_SIZE (MAX_REAL_SIZE * 2 + 1)
#define MAX_unknowns 64
#define MAX_STACK_DEPTH 128

#define UNKNOWN '?'
#define PLUS    '+'
#define MINUS   '-'
#define MULT    '*'
#define DIV     '/'
#define EQ      '='

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))
static void HandleError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
struct Pos { int r, c; };

struct GameState {
    int board[MAX_BOARD_SIZE][MAX_BOARD_SIZE];
    int board_size;
    Pos unknowns_pos[MAX_unknowns];
    int num_unknowns;
    int initial_unknown_values[MAX_unknowns];
    int start_unknown_idx;
};

struct StackNode {
    int values[MAX_unknowns];
    int unknown_idx_to_solve;
    int num_to_try;
};

/*
* ---------------------------------------
*  [    GPU functions (__device__)   ]
* ---------------------------------------
*/
__device__ bool operate(int op, int& result, int n) {
    switch (op) {
    case PLUS: result += n; break;
    case MINUS: result -= n; break;
    case MULT: result *= n; break;
    case DIV:
        if (n != 0 && result % n == 0) result /= n;
        else return false;
        break;
    default: return false;
    }
    return true;
}
__device__ bool is_valid_gpu(const GameState& state, const int* values) {
    int temp_board[MAX_BOARD_SIZE][MAX_BOARD_SIZE];
    for (int i = 0; i < state.board_size; i++) {
        for (int j = 0; j < state.board_size; j++) {
            temp_board[i][j] = state.board[i][j];
        }
    }
    for (int i = 0; i < state.num_unknowns; i++) {
        if (values[i] != UNKNOWN) {
            Pos p = state.unknowns_pos[i];
            temp_board[p.r][p.c] = values[i];
        }
    }
    for (int i = 0; i < state.board_size - 1; i += 2) {
        if (temp_board[i][0] == UNKNOWN) continue;
        int result = temp_board[i][0];
        bool row_complete = true;
        for (int j = 1; j < state.board_size - 2; j += 2) {
            int op = temp_board[i][j];
            int val = temp_board[i][j + 1];
            if (val == UNKNOWN) {
                row_complete = false;
                break;
            }
            if (!operate(op, result, val)) return false;
        }
        if (row_complete && result != temp_board[i][state.board_size - 1]) {
            return false;
        }
    }
    for (int j = 0; j < state.board_size - 1; j += 2) {
        if (temp_board[0][j] == UNKNOWN) continue;
        int result = temp_board[0][j];
        bool col_complete = true;
        for (int i = 1; i < state.board_size - 2; i += 2) {
            int op = temp_board[i][j];
            int val = temp_board[i + 1][j];
            if (val == UNKNOWN) {
                col_complete = false;
                break;
            }
            if (!operate(op, result, val)) return false;
        }
        if (col_complete && result != temp_board[state.board_size - 1][j]) {
            return false;
        }
    }
    return true;
}

__device__ void solve_iterative(const GameState state, int* thread_solution_count_ptr) {
    
    if (state.start_unknown_idx >= state.num_unknowns) {
        if (is_valid_gpu(state, state.initial_unknown_values)) {
            atomicAdd(thread_solution_count_ptr, 1);
        }
        return;
    }

    StackNode stack[MAX_STACK_DEPTH];
    int stack_top = 0;

    stack[stack_top].unknown_idx_to_solve = state.start_unknown_idx;
    stack[stack_top].num_to_try = 0;

    for (int i = 0; i < state.num_unknowns; i++) {
        stack[stack_top].values[i] = state.initial_unknown_values[i];
    }
    stack_top++;

    while (stack_top > 0) {
        stack_top--;
        StackNode current = stack[stack_top];

        if (current.num_to_try > 9 || current.unknown_idx_to_solve >= state.num_unknowns) {
            continue;
        }

        StackNode next_try = current;
        next_try.num_to_try++;
        stack[stack_top] = next_try;
        stack_top++;

        current.values[current.unknown_idx_to_solve] = current.num_to_try;

        if (is_valid_gpu(state, current.values)) {
            if (current.unknown_idx_to_solve == state.num_unknowns - 1) {
                atomicAdd(thread_solution_count_ptr, 1);
            }
            else {
                StackNode next_unknown = current;
                next_unknown.unknown_idx_to_solve++;
                next_unknown.num_to_try = 0;
                stack[stack_top] = next_unknown;
                stack_top++;
            }
        }
    }
}

__global__ void numeric_board_kernel(GameState* subproblems, int num_subproblems, int* solution_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_subproblems) {
        int thread_solution_count = 0;
        solve_iterative(subproblems[idx], &thread_solution_count);
        if (thread_solution_count > 0) {
            atomicAdd(solution_count, thread_solution_count);
        }
    }
}


/*
* ----------------------------
*  [    CPU functions   ]
* ----------------------------
*/
int token_to_int(const std::string& s) {
    if (s == "?") return UNKNOWN;
    if (s == "+") return PLUS;
    if (s == "-") return MINUS;
    if (s == "*") return MULT;
    if (s == "/") return DIV;
    if (s == "=") return EQ;
    return std::stoi(s);
}

bool parse_file(const std::string& filename, GameState& game) {
    std::ifstream infile(filename);
    if (!infile.is_open()) return false;
    int real_size;
    infile >> real_size;
    if (real_size > MAX_REAL_SIZE) return false;
    game.board_size = real_size * 2 + 1;
    game.num_unknowns = 0;
    std::string line;
    std::getline(infile, line);
    for (int i = 0; i < game.board_size - 1; i++) {
        std::getline(infile, line);
        std::stringstream ss(line);
        std::string token;
        if (i % 2 == 0) {
            for (int j = 0; j < game.board_size; j++) {
                ss >> token;
                game.board[i][j] = token_to_int(token);
                if (game.board[i][j] == UNKNOWN) {
                    game.unknowns_pos[game.num_unknowns++] = { i, j };
                }
            }
        }
        else {
            for (int j = 0; j < real_size; j++) {
                ss >> token;
                game.board[i][j * 2] = token_to_int(token);
                if (j * 2 + 1 < game.board_size) game.board[i][j * 2 + 1] = 0;
            }
        }
    }
    std::getline(infile, line);
    std::stringstream ss(line);
    std::string token;
    int last_row_idx = game.board_size - 1;
    for (int j = 0; j < game.board_size; j++) {
        game.board[last_row_idx][j] = 0;
    }
    int col_idx = 0;
    while (ss >> token) {
        if (col_idx < game.board_size) {
            game.board[last_row_idx][col_idx] = token_to_int(token);
        }
        col_idx += 2;
    }
    return true;
}

void show_time(int k, long long milliseconds) {
    double seconds = milliseconds / 1000.0;
    std::cout << "Time for TEST0" << k << ": "
        << std::fixed << std::setprecision(3) << seconds << "s" << std::endl;
}

std::string int_to_symbol(int code) {
    switch (code) {
    case UNKNOWN: return "?";
    case PLUS:    return "+";
    case MINUS:   return "-";
    case MULT:    return "*";
    case DIV:     return "/";
    case EQ:      return "=";
    default:      return std::to_string(code);
    }
}

bool operate_host(int op, int& result, int n) {
    switch (op) {
    case PLUS: result += n; break;
    case MINUS: result -= n; break;
    case MULT: result *= n; break;
    case DIV:
        if (n != 0 && result % n == 0) result /= n;
        else return false;
        break;
    default: return false;
    }
    return true;
}
bool is_valid_host(const GameState& state, const int* values) {
    int temp_board[MAX_BOARD_SIZE][MAX_BOARD_SIZE];
    for (int i = 0; i < state.board_size; i++) {
        for (int j = 0; j < state.board_size; j++) {
            temp_board[i][j] = state.board[i][j];
        }
    }
    for (int i = 0; i < state.num_unknowns; i++) {
        if (values[i] != UNKNOWN) {
            Pos p = state.unknowns_pos[i];
            temp_board[p.r][p.c] = values[i];
        }
    }
    for (int i = 0; i < state.board_size - 1; i += 2) {
        if (temp_board[i][0] == UNKNOWN) continue;
        int result = temp_board[i][0];
        bool row_complete = true;
        for (int j = 1; j < state.board_size - 2; j += 2) {
            int op = temp_board[i][j];
            int val = temp_board[i][j + 1];
            if (val == UNKNOWN) {
                row_complete = false;
                break;
            }
            if (!operate_host(op, result, val)) return false;
        }
        if (row_complete && result != temp_board[i][state.board_size - 1]) {
            return false;
        }
    }
    for (int j = 0; j < state.board_size - 1; j += 2) {
        if (temp_board[0][j] == UNKNOWN) continue;
        int result = temp_board[0][j];
        bool col_complete = true;
        for (int i = 1; i < state.board_size - 2; i += 2) {
            int op = temp_board[i][j];
            int val = temp_board[i + 1][j];
            if (val == UNKNOWN) {
                col_complete = false;
                break;
            }
            if (!operate_host(op, result, val)) return false;
        }
        if (col_complete && result != temp_board[state.board_size - 1][j]) {
            return false;
        }
    }
    return true;
}

void generate_subproblems_recursive_host(
    const GameState& base_initial_game,
    std::vector<GameState>& subproblems_out,
    std::vector<int>& current_branch_values,
    int current_unknown_idx_to_branch,
    const int N_BRANCHING_LEVELS
) {
    if (current_unknown_idx_to_branch == N_BRANCHING_LEVELS || current_unknown_idx_to_branch == base_initial_game.num_unknowns) {
        GameState subproblem = base_initial_game;
        for (int i = 0; i < base_initial_game.num_unknowns; ++i) {
            subproblem.initial_unknown_values[i] = current_branch_values[i];
        }
        subproblem.start_unknown_idx = current_unknown_idx_to_branch;
        subproblems_out.push_back(subproblem);
        return;
    }
    for (int val = 0; val <= 9; ++val) {
        current_branch_values[current_unknown_idx_to_branch] = val;
        std::vector<int> temp_values_for_check(base_initial_game.num_unknowns);
        for (int i = 0; i <= current_unknown_idx_to_branch; ++i) {
            temp_values_for_check[i] = current_branch_values[i];
        }
        for (int i = current_unknown_idx_to_branch + 1; i < base_initial_game.num_unknowns; ++i) {
            temp_values_for_check[i] = UNKNOWN;
        }
        if (is_valid_host(base_initial_game, temp_values_for_check.data())) {
            generate_subproblems_recursive_host(
                base_initial_game,
                subproblems_out,
                current_branch_values,
                current_unknown_idx_to_branch + 1,
                N_BRANCHING_LEVELS
            );
        }
    }
    current_branch_values[current_unknown_idx_to_branch] = UNKNOWN;
}

void generate_subproblems(const GameState& initial_game, std::vector<GameState>& subproblems_out) {
    int N_BRANCHING_LEVELS = 4;
    if (initial_game.num_unknowns == 0) {
        N_BRANCHING_LEVELS = 0;
    }
    else if (initial_game.num_unknowns < N_BRANCHING_LEVELS) {
        N_BRANCHING_LEVELS = initial_game.num_unknowns;
    }
    subproblems_out.clear();
    std::vector<int> current_branch_values(initial_game.num_unknowns, UNKNOWN);
    GameState base_game_for_recursive = initial_game;
    for (int i = 0; i < base_game_for_recursive.num_unknowns; ++i) {
        base_game_for_recursive.initial_unknown_values[i] = UNKNOWN;
    }
    base_game_for_recursive.start_unknown_idx = 0;
    generate_subproblems_recursive_host(
        base_game_for_recursive,
        subproblems_out,
        current_branch_values,
        0,
        N_BRANCHING_LEVELS
    );
    std::cout << "Se generaron " << subproblems_out.size() << " subproblemas en la CPU." << std::endl;
}

int main() {
    const int MIN_TEST = 0; const int MAX_TEST = 6;
    for (int k = MIN_TEST; k <= MAX_TEST; k++) {
        std::string filename = "Test/test0" + std::to_string(k) + ".txt";
        GameState initial_game = {};
        if (!parse_file(filename, initial_game)) {
            std::cerr << "Warning: No se pudo abrir o analizar el archivo " << filename << ". Saltando." << std::endl;
            continue;
        }
        auto start_time = std::chrono::high_resolution_clock::now();
        std::vector<GameState> host_subproblems;
        generate_subproblems(initial_game, host_subproblems);
        int num_subproblems = host_subproblems.size();
        if (num_subproblems == 0) {
            std::cout << "Prueba " << k << ": Se encontraron 0 soluciones en 0 subproblemas." << std::endl;
            show_time(k, 0);
            continue;
        }
        GameState* d_subproblems;
        int* d_solution_count;
        HANDLE_ERROR(cudaMalloc((void**)&d_subproblems, num_subproblems * sizeof(GameState)));
        HANDLE_ERROR(cudaMalloc((void**)&d_solution_count, sizeof(int)));
        HANDLE_ERROR(cudaMemcpy(d_subproblems, host_subproblems.data(), num_subproblems * sizeof(GameState), cudaMemcpyHostToDevice));
        int zero = 0;
        HANDLE_ERROR(cudaMemcpy(d_solution_count, &zero, sizeof(int), cudaMemcpyHostToDevice));
        const int BLOCK_SIZE = 256;
        int num_blocks = (num_subproblems + BLOCK_SIZE - 1) / BLOCK_SIZE;
        numeric_board_kernel << <num_blocks, BLOCK_SIZE >> > (d_subproblems, num_subproblems, d_solution_count);
        HANDLE_ERROR(cudaDeviceSynchronize());
        int total_solutions = 0;
        HANDLE_ERROR(cudaMemcpy(&total_solutions, d_solution_count, sizeof(int), cudaMemcpyDeviceToHost));
        std::cout << "Prueba " << k << ": Se encontraron " << total_solutions << " soluciones en "
            << num_subproblems << " subproblemas." << std::endl;
        HANDLE_ERROR(cudaFree(d_subproblems));
        HANDLE_ERROR(cudaFree(d_solution_count));
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        show_time(k, duration_ms.count());
    }
    return 0;
}