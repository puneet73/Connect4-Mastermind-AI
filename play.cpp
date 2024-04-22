#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>

const int BOARD_WIDTH = 7;
const int BOARD_HEIGHT = 6;
const int EMPTY = 0;
const int PLAYER1 = 1;
const int PLAYER2 = 2;
const double C_PUCT = 1.0;
const int NUM_SIMULATIONS = 1000;

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
            exploitation_score = 0.1; // Assign a default value (or a small positive value)
        }

        double exploration_score = 0.0;
        if (children[i]->visit_count > 0) {
            exploration_score = std::sqrt(2 * std::log(visit_count) / children[i]->visit_count);
        } else {
            exploration_score = 0.1; // Assign a default value (or a small positive value)
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
    return child;
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

bool checkDraw(const std::vector<int>& board) {
    for (int col = 0; col < BOARD_WIDTH; col++) {
        if (findFirstEmptyRow(board, col) != -1) return false;
    }
    return true;
}

std::vector<int> mcts(Node* root) {
    if (root == nullptr || root->board.empty()) {
        return std::vector<int>(BOARD_WIDTH * BOARD_HEIGHT, EMPTY);
    }
    for (int i = 0; i < NUM_SIMULATIONS; i++) {
        Node* node = root;
        // Selection step - until we hit a node that can be expanded or is terminal
        while(node->selectChild() != nullptr) {
            node = node->selectChild();
        }

        // Here, `node` is either a leaf to be expanded (if game not ended) or a terminal game state
         if (!checkWin(node->board, PLAYER1) && !checkWin(node->board, PLAYER2) && !checkDraw(node->board)) { 
            // Check for valid moves (actions)
            std::vector<int> available_moves;
            for (int i = 0; i < BOARD_WIDTH; i++) {
                if (findFirstEmptyRow(node->board, i) != -1) {
                    available_moves.push_back(i);
                }
            }

            // If there are valid moves, choose one randomly to expand
            if (!available_moves.empty()) {
                int random_index = std::rand() % available_moves.size();
                int action = available_moves[random_index];
                Node* new_child = node->expand(action);  // Create the new child node
                node = new_child;                       // Set the current node to the new child for rollout
            }
        }
        
        // Rollout from the selected or expanded node
        double reward = node->rollout();

        // Backpropagation of the rollout result
        node->backpropagate(reward);
    }

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

        // AI (Player 2) move
        root->board = mcts(root);
        if (!checkWin(root->board, PLAYER2) && !checkDraw(root->board)) {
            root = root->selectChild();
        }

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