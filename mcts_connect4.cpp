#include <bits/stdc++.h>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include <omp.h>
#include <chrono>
#include <iostream>

const int BOARD_WIDTH = 7;
const int BOARD_HEIGHT = 6;
const int EMPTY = 0;
const int PLAYER1 = 1;
const int PLAYER2 = 2;
const double C_PUCT = 1.0;
const int NUM_SIMULATIONS = 10000;
const int NUM_ITERATIONS = 10000;

class Node {
public:
    std::vector<int> board;
    int player;
    std::vector<Node*> children;
    int visit_count;
    double total_reward;

    Node(const std::vector<int>& board, int player);
    ~Node() {
        for (Node* child : children) {
            if (child) {
                delete child;
            }
        }
    }
    Node* selectChild();
    Node* expand(int action);
    double rollout();
    void backpropagate(double reward);
    void backpropagateParallel(double reward); // Parallel version of backpropagate

    Node* parent = nullptr;
};

int findFirstEmptyRow(const std::vector<int>& board, int column);
bool checkWin(const std::vector<int>& board, int player);
bool checkDraw(const std::vector<int>& board);
std::vector<int> mcts(Node* root);
void printBoard(const std::vector<int>& board);

Node::Node(const std::vector<int>& board, int player) : board(board), player(player), visit_count(0), total_reward(0) {
    children.resize(BOARD_WIDTH, nullptr);
}

Node* Node::selectChild() {
    Node* best_child = nullptr;
    double best_score = -std::numeric_limits<double>::infinity();

    for (int i = 0; i < children.size(); i++) {
        if (children[i] == nullptr) continue;

        double exploitation_score = 0.0;
        if (children[i]->visit_count > 0) {
            exploitation_score = children[i]->total_reward / children[i]->visit_count;
        } else {
            exploitation_score = 0.00001; // Assign a default value (or a small positive value)
        }

        double exploration_score = 0.0;
        if (children[i]->visit_count > 0) {
            exploration_score = std::sqrt(2 * std::log(visit_count) / children[i]->visit_count);
        } else {
            exploration_score = 0.00001; // Assign a default value (or a small positive value)
        }

        double score = exploitation_score + C_PUCT * exploration_score;

        if (score > best_score) {
            best_score = score;
            best_child = children[i];
        }
    }

    return best_child;
}

Node* Node::expand(int action) {
    std::vector<int> new_board(board);
    int row = findFirstEmptyRow(new_board, action);
    new_board[row * BOARD_WIDTH + action] = player;
    int new_player = (player == PLAYER1) ? PLAYER2 : PLAYER1;
    Node* child = new Node(new_board, new_player);
    child->parent = this;  // Set the parent of the child node
    children[action] = child;

    if (checkWin(new_board, player)) {
        backpropagate(1.0); 
    } else {
        for (int dir : {-1, 0, 1}) { 
            int consecutive = 0;
            int consecutiveOpponent = 0;
            int maxConsecutive = 0;
            for (int i = -3; i <= 3; i++) {
                int r = row + dir * i;
                int c = action + dir * i;
                if (r >= 0 && r < BOARD_HEIGHT && c >= 0 && c < BOARD_WIDTH) {
                    if (new_board[r * BOARD_WIDTH + c] == player) {
                        consecutive++;
                        consecutiveOpponent = 0;
                    } else if (new_board[r * BOARD_WIDTH + c] == (player == PLAYER1 ? PLAYER2 : PLAYER1)) {
                        consecutiveOpponent++;
                        consecutive = 0;
                    } else {
                        // Empty cell, reset counters
                        maxConsecutive = std::max(maxConsecutive, consecutive);
                        consecutive = 0;
                        consecutiveOpponent = 0;
                    }
                }
                if (consecutive == 3) {
                    backpropagate(0.8); 
                    break;
                }
                if (consecutiveOpponent == 3) {
                    backpropagate(-0.6); 
                    break;
                }
            }
        }
    }
    return child;
}

int evaluateImmediateWin(Node* node) {
    if (checkWin(node->board, node->player)) {
        return 1000; // Large score for winning move
    } else if (checkWin(node->board, node->player == PLAYER1 ? PLAYER2 : PLAYER1)) {
        return 1001; // Smaller score, but still significant, for blocking opponent win
    }
    return 0; // No immediate win or block
}


int bfsImmediateAnalysis(Node* root, int depth) {
    struct Move {
        Node* node;
        int depth;
        int column;  // Move that led to this node
    };

    std::queue<Move> queue;
    Move bestMove = {nullptr, 0, -1};
    int bestScore = std::numeric_limits<int>::min();

    // Enqueue initial moves
    for (int i = 0; i < BOARD_WIDTH; i++) {
        if (findFirstEmptyRow(root->board, i) != -1) {
            Node* child = root->expand(i);
            queue.push({child, 1, i});
        }
    }

    while (!queue.empty()) {
        Move currentMove = queue.front();
        queue.pop();

        // Score this move
        int score = evaluateImmediateWin(currentMove.node);
        if (score > bestScore) {
            bestScore = score;
            bestMove = currentMove;
        }

        if (currentMove.depth < depth) {
            for (int i = 0; i < BOARD_WIDTH; i++) {
                if (findFirstEmptyRow(currentMove.node->board, i) != -1) {
                    Node* child = currentMove.node->expand(i);
                    queue.push({child, currentMove.depth + 1, currentMove.column});
                }
            }
        }
    }

    while (!queue.empty()) {
        delete queue.front().node;
        queue.pop();
    }

    return bestMove.column;
}


double Node::rollout() {
    std::vector<int> state(board);
    int player = this->player;

    while (true) {
        if (checkWin(state, PLAYER1)) return 1.0;
        if (checkWin(state, PLAYER2)) return -1.0;
        if (checkDraw(state)) return 0.0;

        std::vector<int> available_moves;
        for (int i = 0; i < BOARD_WIDTH; i++) {
            if (findFirstEmptyRow(state, i) != -1) available_moves.push_back(i);
        }
        int action = available_moves[std::rand() % available_moves.size()];
        int row = findFirstEmptyRow(state, action);
        state[row * BOARD_WIDTH + action] = player;
        player = (player == PLAYER1) ? PLAYER2 : PLAYER1;
    }
}

void Node::backpropagate(double reward) {
    visit_count++;
    total_reward += reward;
    if (parent) parent->backpropagate(-reward);
}

void Node::backpropagateParallel(double reward) {
    visit_count++;
    total_reward += reward;

    if (parent) {
        if (parent->parent != nullptr) { // Not the root node
            parent->backpropagateParallel(-reward); 
        } else { // Parent is the root node
            #pragma omp critical
            {
                parent->visit_count++;
                parent->total_reward += -reward;
            }
        }
    }
}

int findFirstEmptyRow(const std::vector<int>& board, int column) {
    for (int row = BOARD_HEIGHT - 1; row >= 0; row--) {
        if (board[row * BOARD_WIDTH + column] == EMPTY) return row;
    }
    return -1;
}

bool checkWin(const std::vector<int>& board, int player) {
    // Check rows
    for (int row = 0; row < BOARD_HEIGHT; row++) {
        for (int col = 0; col <= BOARD_WIDTH - 4; col++) {
            if (board[row * BOARD_WIDTH + col] == player &&
                board[row * BOARD_WIDTH + col + 1] == player &&
                board[row * BOARD_WIDTH + col + 2] == player &&
                board[row * BOARD_WIDTH + col + 3] == player)
                return true;
        }
    }
    // Check columns
    for (int col = 0; col < BOARD_WIDTH; col++) {
        for (int row = 0; row <= BOARD_HEIGHT - 4; row++) {
            if (board[row * BOARD_WIDTH + col] == player &&
                board[(row + 1) * BOARD_WIDTH + col] == player &&
                board[(row + 2) * BOARD_WIDTH + col] == player &&
                board[(row + 3) * BOARD_WIDTH + col] == player)
                return true;
        }
    }
    // Check diagonals
    for (int row = 0; row <= BOARD_HEIGHT - 4; row++) {
        for (int col = 0; col <= BOARD_WIDTH - 4; col++) {
            if (board[row * BOARD_WIDTH + col] == player &&
                board[(row + 1) * BOARD_WIDTH + col + 1] == player &&
                board[(row + 2) * BOARD_WIDTH + col + 2] == player &&
                board[(row + 3) * BOARD_WIDTH + col + 3] == player)
                return true;
            if (board[row * BOARD_WIDTH + col + 3] == player &&
                board[(row + 1) * BOARD_WIDTH + col + 2] == player &&
                board[(row + 2) * BOARD_WIDTH + col + 1] == player &&
                board[(row + 3) * BOARD_WIDTH + col] == player)
                return true;
        }
    }
    return false;
}

bool checkWinParallel(const std::vector<int>& board, int player) {
    bool winFound = false;
    #pragma omp parallel for
    for (int row = 0; row < BOARD_HEIGHT; row++) {
        if (winFound) continue; // If a win is already found, other threads can stop
        for (int col = 0; col <= BOARD_WIDTH - 4; col++) {
            if (board[row * BOARD_WIDTH + col] == player &&
                board[row * BOARD_WIDTH + col + 1] == player &&
                board[row * BOARD_WIDTH + col + 2] == player &&
                board[row * BOARD_WIDTH + col + 3] == player) {
                #pragma omp critical
                {
                    winFound = true; 
                }
            }
        }
    }

    #pragma omp parallel for 
    for (int col = 0; col < BOARD_WIDTH; col++) {
        if (winFound) continue; 
        for (int row = 0; row <= BOARD_HEIGHT - 4; row++) {
            if (board[row * BOARD_WIDTH + col] == player &&
                board[(row + 1) * BOARD_WIDTH + col] == player &&
                board[(row + 2) * BOARD_WIDTH + col] == player &&
                board[(row + 3) * BOARD_WIDTH + col] == player) {
                #pragma omp critical
                {
                    winFound = true; 
                }
            }
        }
    }

    #pragma omp parallel for 
    for (int row = 0; row <= BOARD_HEIGHT - 4; row++) {
        if (winFound) continue; 
        for (int col = 0; col <= BOARD_WIDTH - 4; col++) {
            if ((board[row * BOARD_WIDTH + col] == player &&
                 board[(row + 1) * BOARD_WIDTH + col + 1] == player &&
                 board[(row + 2) * BOARD_WIDTH + col + 2] == player &&
                 board[(row + 3) * BOARD_WIDTH + col + 3] == player) || 
                (board[row * BOARD_WIDTH + col + 3] == player &&
                 board[(row + 1) * BOARD_WIDTH + col + 2] == player &&
                 board[(row + 2) * BOARD_WIDTH + col + 1] == player &&
                 board[(row + 3) * BOARD_WIDTH + col] == player)) {
                #pragma omp critical
                {
                    winFound = true; 
                }
            }
        }
    }
    return winFound;
}

bool checkDraw(const std::vector<int>& board) {
    for (int col = 0; col < BOARD_WIDTH; col++) {
        if (findFirstEmptyRow(board, col) != -1) return false;
    }
    return true;
}

bool checkDrawParallel(const std::vector<int>& board) {
    bool isDraw = true; // Shared variable
    #pragma omp parallel for 
    for (int col = 0; col < BOARD_WIDTH; col++) {
        if (findFirstEmptyRow(board, col) != -1) {
            #pragma omp critical
            {
                isDraw = false; 
            }
        }
    }
    return isDraw;
}

int countWinningLines(const std::vector<int>& board, int player) {
    int count = 0;

    // Check rows
    for (int row = 0; row < BOARD_HEIGHT; row++) {
        for (int col = 0; col <= BOARD_WIDTH - 4; col++) {
            if (board[row * BOARD_WIDTH + col] == EMPTY || board[row * BOARD_WIDTH + col] == player) {
                if (board[row * BOARD_WIDTH + col + 1] == player && 
                    board[row * BOARD_WIDTH + col + 2] == player &&
                    board[row * BOARD_WIDTH + col + 3] == player) {
                    count++; 
                }
            }
        }
    }

    // Check columns
    for (int col = 0; col < BOARD_WIDTH; col++) {
        for (int row = 0; row <= BOARD_HEIGHT - 4; row++) {
            if (board[row * BOARD_WIDTH + col] == EMPTY || board[row * BOARD_WIDTH + col] == player) {
                if (board[(row + 1) * BOARD_WIDTH + col] == player &&
                    board[(row + 2) * BOARD_WIDTH + col] == player &&
                    board[(row + 3) * BOARD_WIDTH + col] == player) {
                    count++;
                }
            }
        }
    }

    // Check diagonals (top-left to bottom-right)
    for (int row = 0; row <= BOARD_HEIGHT - 4; row++) {
        for (int col = 0; col <= BOARD_WIDTH - 4; col++) {
            if (board[row * BOARD_WIDTH + col] == EMPTY || board[row * BOARD_WIDTH + col] == player) {
                if (board[(row + 1) * BOARD_WIDTH + col + 1] == player &&
                    board[(row + 2) * BOARD_WIDTH + col + 2] == player &&
                    board[(row + 3) * BOARD_WIDTH + col + 3] == player) {
                    count++;
                }
            }
        }
    }

    // Check diagonals (bottom-left to top-right)
    for (int row = 3; row < BOARD_HEIGHT; row++) { 
        for (int col = 0; col <= BOARD_WIDTH - 4; col++) {
            if (board[row * BOARD_WIDTH + col] == EMPTY || board[row * BOARD_WIDTH + col] == player) {
                if (board[(row - 1) * BOARD_WIDTH + col + 1] == player &&
                    board[(row - 2) * BOARD_WIDTH + col + 2] == player &&
                    board[(row - 3) * BOARD_WIDTH + col + 3] == player) {
                    count++;
                }
            }
        }
    }

    return count;
}

int countCenterColumnPieces(const std::vector<int>& board, int player) {
    int count = 0;
    int centerCol = BOARD_WIDTH / 2; 

    for (int row = 0; row < BOARD_HEIGHT; row++) {
        if (board[row * BOARD_WIDTH + centerCol] == player) {
            count++;
        }
    }
    return count;
}

int evaluateHeuristic(Node* node) {
    int player = node->player;
    int opponent = (player == PLAYER1) ? PLAYER2 : PLAYER1;

    // 1. Check for immediate win:
    if (checkWin(node->board, player)) {
        return 1000; 
    } else if (checkWin(node->board, opponent)) {
        return -1000; 
    }

    // 2. Count potential winning lines:
    int playerLines = countWinningLines(node->board, player);
    int opponentLines = countWinningLines(node->board, opponent);

    // 3. Count pieces in the center column
    int playerCenterPieces = countCenterColumnPieces(node->board, player);
    int opponentCenterPieces = countCenterColumnPieces(node->board, opponent); 

    // 4. Calculate the heuristic score:
    int score = (playerLines - opponentLines) * 10 +  // Winning lines are important
                (playerCenterPieces - opponentCenterPieces); // Center control is valuable

    return score;
}


std::vector<int> mcts(Node* root) {
    if (root == nullptr || root->board.empty()) {
        return std::vector<int>(BOARD_WIDTH * BOARD_HEIGHT, EMPTY);
    }

    for (int i = 0; i < NUM_SIMULATIONS; i++) {
        Node* node = root;
        
        // Selection
        while (node->selectChild() != nullptr) { 
            node = node->selectChild();
        }

        // Expansion - Expand on all valid children 
        std::vector<int> available_moves;
        for (int col = 0; col < BOARD_WIDTH; col++) {
            if (findFirstEmptyRow(node->board, col) != -1) {
                available_moves.push_back(col);
            }
        }
        for (int move : available_moves) {
            node->expand(move); // Expand on all valid moves
        }

        // Simulation
        double reward = 0.0;
        if (node->selectChild() != nullptr) { // If we expanded, choose a child to simulate from
            node = node->selectChild();
            reward = node->rollout(); 
        } else { // If no expansion was possible (terminal node), evaluate it
            reward = evaluateHeuristic(node);
        }

        // Backpropagation
        node->backpropagate(reward); 
    }

    // Select the best move based on visit count
    Node* best_child = nullptr;
    int best_visit_count = 0;
    for (Node* child : root->children) {
        if (child != nullptr && child->visit_count > best_visit_count) {
            best_child = child;
            best_visit_count = child->visit_count;
        }
    }
    return best_child ? best_child->board : std::vector<int>(BOARD_WIDTH * BOARD_HEIGHT, EMPTY);
}

std::vector<int> mcts_parallel_1(Node* root) {
    if (root == nullptr || root->board.empty()) {
        return std::vector<int>(BOARD_WIDTH * BOARD_HEIGHT, EMPTY);
    }

    #pragma omp parallel for
    for (int i = 0; i < NUM_SIMULATIONS; i++) {
        Node* node = root;

        // Selection
        while (node->selectChild() != nullptr) { 
            node = node->selectChild();
        }

        // Expansion
        std::vector<int> available_moves;
        for (int col = 0; col < BOARD_WIDTH; col++) {
            if (findFirstEmptyRow(node->board, col) != -1) {
                available_moves.push_back(col);
            }
        }

        #pragma omp critical
        {
            for (int move : available_moves) {
                node->expand(move); 
            }
        }

        // Simulation
        double reward = 0.0;
        if (node->selectChild() != nullptr) { 
            node = node->selectChild();
            reward = node->rollout(); 
        } else { 
            reward = evaluateHeuristic(node);
        }

        // Backpropagation
        #pragma omp critical 
        {
            node->backpropagate(reward); 
        }
    }

    // Select the best move based on visit count
    Node* best_child = nullptr;
    int best_visit_count = 0;
    for (Node* child : root->children) {
        if (child != nullptr && child->visit_count > best_visit_count) {
            best_child = child;
            best_visit_count = child->visit_count;
        }
    }
    return best_child ? best_child->board : std::vector<int>(BOARD_WIDTH * BOARD_HEIGHT, EMPTY);
}

std::vector<int> mcts_parallel_2(Node* root) {
    if (root == nullptr || root->board.empty()) {
        return std::vector<int>(BOARD_WIDTH * BOARD_HEIGHT, EMPTY);
    }

    // Initialize root's children based on available moves
    std::vector<int> available_moves;
    for (int col = 0; col < BOARD_WIDTH; col++) {
        if (findFirstEmptyRow(root->board, col) != -1) {
            available_moves.push_back(col);
            root->expand(col); // Expand on all valid moves
        }
    }

    // Parallel MCTS for each child of the root
    #pragma omp parallel for
    for (int i = 0; i < root->children.size(); ++i) {
        if (root->children[i] != nullptr) {
            for (int j = 0; j < NUM_SIMULATIONS / root->children.size(); ++j) { // Distribute simulations evenly
                Node* node = root->children[i];

                // Selection
                while (node->selectChild() != nullptr) {
                    node = node->selectChild();
                }

                // Expansion (if not a terminal node)
                std::vector<int> child_moves;
                for (int col = 0; col < BOARD_WIDTH; col++) {
                    if (findFirstEmptyRow(node->board, col) != -1) {
                        child_moves.push_back(col);
                        node->expand(col); // Expand on all valid moves
                    }
                }

                // Simulation
                double reward = 0.0;
                if (node->selectChild() != nullptr) { 
                    node = node->selectChild();
                    reward = node->rollout();
                } else { 
                    reward = evaluateHeuristic(node);
                }

                // Backpropagation (modified for parallel subtrees)
                node->backpropagateParallel(reward);
            }
        }
    }

    // Select the best move based on visit count
    Node* best_child = nullptr;
    int best_visit_count = 0;
    for (Node* child : root->children) {
        if (child != nullptr && child->visit_count > best_visit_count) {
            best_child = child;
            best_visit_count = child->visit_count;
        }
    }
    return best_child ? best_child->board : std::vector<int>(BOARD_WIDTH * BOARD_HEIGHT, EMPTY);
}

void printBoard(const std::vector<int>& board) {
    std::cout << "-------------" << std::endl;
    for (int row = 0; row < BOARD_HEIGHT; row++) {
        std::cout << "| ";
        for (int col = 0; col < BOARD_WIDTH; col++) {
            int index = row * BOARD_WIDTH + col;
            if (board[index] == EMPTY) {
                std::cout << "\033[0m" << "." << "\033[0m" << " | ";
            } else if (board[index] == PLAYER1) {
                std::cout << "\033[1;34m" << "X" << "\033[0m" << " | ";  // Print Player 1 pieces in blue
            } else {
                std::cout << "\033[1;31m" << "O" << "\033[0m" << " | ";  // Print Player 2 pieces in red
            }
        }
        std::cout << std::endl;
    }
    std::cout << "-------------" << std::endl;
}

int main() {
    std::vector<int> initial_board(BOARD_WIDTH * BOARD_HEIGHT, EMPTY);
    Node* root = new Node(initial_board, PLAYER1);

    while (true) {
        printBoard(root->board);

        // Player 1 move
        std::vector<int> available_moves;
        for (int i = 0; i < BOARD_WIDTH; i++) {
            if (findFirstEmptyRow(root->board, i) != -1) available_moves.push_back(i);
        }
        int player1_move;
        std::cout << "Player 1, enter your move (0-6): ";
        std::cin >> player1_move;
        if (std::find(available_moves.begin(), available_moves.end(), player1_move) == available_moves.end()) {
            std::cout << "Invalid move, try again." << std::endl;
            continue;
        }
        root = root->expand(player1_move);

        

        // Check for win or draw
        if (checkWin(root->board, PLAYER1)) {
            printBoard(root->board);
            std::cout << "Player 1 wins!" << std::endl;
            break;
        } else if (checkWin(root->board, PLAYER2)) {
            printBoard(root->board);
            std::cout << "Player 2 wins!" << std::endl;
            break;
        } else if (checkDraw(root->board)) {
            printBoard(root->board);
            std::cout << "It's a draw!" << std::endl;
            break;
        }

        // auto result_start = std::chrono::high_resolution_clock::now();
        // for (int i = 0; i < NUM_ITERATIONS; ++i) {
        //     checkWin(root->board, PLAYER1);
        // }
        // auto result_end = std::chrono::high_resolution_clock::now();
        // std::chrono::nanoseconds elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(result_end - result_start);
        // std::cout << "Time taken for checkWin serial (" << NUM_ITERATIONS << " iterations): " << elapsed_ns.count() << " ns" << std::endl;

        // result_start = std::chrono::high_resolution_clock::now();
        // for (int i = 0; i < NUM_ITERATIONS; ++i) {
        //     checkDraw(root->board);
        // }
        // result_end = std::chrono::high_resolution_clock::now();
        // elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(result_end - result_start);
        // std::cout << "Time taken for checkDraw serial (" << NUM_ITERATIONS << " iterations): " << elapsed_ns.count() << " ns" << std::endl;

        // result_start = std::chrono::high_resolution_clock::now();
        // for (int i = 0; i < NUM_ITERATIONS; ++i) {
        //     checkWinParallel(root->board, PLAYER1);
        // }
        // result_end = std::chrono::high_resolution_clock::now();
        // elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(result_end - result_start);
        // std::cout << "Time taken for checkWin parallel (" << NUM_ITERATIONS << " iterations): " << elapsed_ns.count() << " ns" << std::endl;

        // result_start = std::chrono::high_resolution_clock::now();
        // for (int i = 0; i < NUM_ITERATIONS; ++i) {
        //     checkDrawParallel(root->board);
        // }
        // result_end = std::chrono::high_resolution_clock::now();
        // elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(result_end - result_start);
        // std::cout << "Time taken for checkDraw parallel (" << NUM_ITERATIONS << " iterations): " << elapsed_ns.count() << " ns" << std::endl;


        // AI (Player 2) move

        //benchmarking mcts serial, parallel approach 1 and parallel approach 2
        auto result_start = std::chrono::high_resolution_clock::now();
        mcts(root);
        auto result_end = std::chrono::high_resolution_clock::now();
        std::chrono::nanoseconds elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(result_end - result_start);
        std::cout << "Time taken for mcts serial: " << elapsed_ns.count() << " ns" << std::endl;

        result_start = std::chrono::high_resolution_clock::now();
        mcts_parallel_1(root);
        result_end = std::chrono::high_resolution_clock::now();
        elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(result_end - result_start);
        std::cout << "Time taken for mcts parallel approach 1: " << elapsed_ns.count() << " ns" << std::endl;

        result_start = std::chrono::high_resolution_clock::now();
        mcts_parallel_2(root);
        result_end = std::chrono::high_resolution_clock::now();
        elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(result_end - result_start);
        std::cout << "Time taken for mcts parallel approach 2: " << elapsed_ns.count() << " ns" << std::endl;


        root->board = mcts_parallel_2(root);
        if (!checkWin(root->board, PLAYER2) && !checkDraw(root->board)) {
            root = root->selectChild();
        }
        // // AI (Player 2) move using BFS analysis for immediate moves
        // int bestMove = bfsImmediateAnalysis(root, 4); // Checking up to 3 moves ahead
        // if (bestMove != -1) {
        //     root = root->expand(bestMove);
        // } else {
        //     // Fall back to MCTS if no immediate beneficial move is found
        //     root->board = mcts(root);
        //     root = root->selectChild();
        // }


        // Check for win or draw
        if (checkWin(root->board, PLAYER1)) {
            printBoard(root->board);
            std::cout << "Player 1 wins!" << std::endl;
            break;
        } else if (checkWin(root->board, PLAYER2)) {
            printBoard(root->board);
            std::cout << "Player 2 wins!" << std::endl;
            break;
        } else if (checkDraw(root->board)) {
            printBoard(root->board);
            std::cout << "It's a draw!" << std::endl;
            break;
        }
    }

    delete root;

    return 0;
}