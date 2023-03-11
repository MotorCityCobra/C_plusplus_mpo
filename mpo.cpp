#include <fstream>
#include <iterator>
#include <string>
#include <fmt/core.h>
#include <filesystem>

#include <torch/torch.h>
#include <torch/script.h>
#include <ATen/Tensor.h>

using namespace torch::indexing;
#include "GLFW/glfw3.h"

#include "mujoco/mujoco.h"
#include "NumCpp.hpp"
#include <vector>
#include <random>

torch::Device device(torch::kCUDA);

// torch::jit::script::Module module = torch::jit::load(s_model_name, torch::kCUDA);
// const std::string device_string = "cuda:2";
// module.to(device_string);


std::random_device rd;     // Only used once to initialise (seed) engine
std::mt19937 rng(rd());    // Random-number engine used (Mersenne-Twister in this case)

const int maxthread = 512;

std::uniform_int_distribution<std::mt19937::result_type> randk(0, 1000); // distribution in range [1, 6]

// MuJoCo data structures
mjModel* m = NULL;                  // MuJoCo model
mjData* d[maxthread];                   // MuJoCo data

mjtSensor* s = NULL;                // Sensor data
mjvCamera cam;                      // abstract camera
mjvOption opt;                      // visualization options
mjvScene scn;                       // abstract scene
mjrContext con;                     // custom GPU context

        // Mod is stored episodes * steps in episode / batch size sampled per step

int episode_and_training_batch = 125;
int batch_multiple = 4;
int B_size = batch_multiple * episode_and_training_batch;
int n_actions = 17;
int n_obs_numbers = 385;

char filename[] = "../humanoidstandup.xml";

// deallocate and print message
int finish(const char* msg = NULL, mjModel* m = NULL) {
  // deallocate model
  if (m) {
    mj_deleteModel(m);
  }
  // print message
  if (msg) {
    std::printf("%s\n", msg);
  }
  return 0;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> gaussian_kl( \
    torch::Tensor μi, torch::Tensor μ, torch::Tensor Ai, torch::Tensor A) {

    auto n = A.size(-1);
    μi = μi.unsqueeze(-1);
    μ = μ.unsqueeze(-1);    
    auto Σi = torch::matmul(Ai, Ai.transpose(-2, -1));  // (B, n, n)
    auto Σ = torch::matmul(A, A.transpose(-2, -1));  // (B, n, n)

    auto Σi_det = torch::clamp_min(Σi.det(), 1e-6);
    auto Σ_det = torch::clamp_min(Σ.det(), 1e-6);
    Σi_det = torch::nan_to_num(Σi_det);
    Σ_det = torch::nan_to_num(Σ_det);
    auto Σi_inv = Σi.inverse();  // (B, n, n)
    auto Σ_inv = Σ.inverse();  // (B, n, n)

    auto inner_μ = torch::matmul((μ - μi).transpose(-2, -1), torch::matmul(Σi_inv, \ 
                                                    (μ - μi))).squeeze(); //(B,)
    auto inner_Σ = torch::log(Σ_det / Σi_det) - n + torch::matmul(Σ_inv, Σi).diagonal(\
                                                    0, -2, -1).sum(-1); //(B,)
    auto C_μ = 0.5 * torch::mean(inner_μ);
    auto C_Σ = 0.5 * torch::mean(inner_Σ);

    return {torch::nan_to_num(C_μ), torch::nan_to_num(C_Σ), torch::mean(Σi_det), torch::mean(Σ_det)};
}

torch::Tensor MultivariateNormal_sampler(int actions, const torch::Tensor mean, const torch::Tensor cholesky) {
    auto many_actions_for_state = torch::tensor({}).to(device);
    torch::Tensor rand_cholesky;
    torch::Tensor eps;
    for (int z=0; z<actions; z++) {
        eps = torch::randn(cholesky[0].sizes()[0]).to(device);
        rand_cholesky = mean + torch::matmul(cholesky, eps).squeeze(-1);
        many_actions_for_state = torch::cat({(rand_cholesky).unsqueeze(0), many_actions_for_state}, 0);
    }
    return many_actions_for_state;
}

torch::Tensor mahalanobis_distance(torch::Tensor x, torch::Tensor mean, torch::Tensor chol) {
    // calculate the Mahalanobis Distance
    auto diff = x - mean.expand({x.size(0), -1, -1});
    auto matmul_one = torch::linalg::solve_triangular(chol.expand({x.size(0), -1, -1, -1}), diff.unsqueeze(3), false, true, false);
    // auto md = torch::matmul(matmul_one, matmul_one).squeeze(-1);
    auto md = torch::matmul(matmul_one.permute({0, 1, 3, 2}), matmul_one);
    return torch::nan_to_num(md.squeeze(-1).squeeze(-1));
}

torch::Tensor log_prob_mvn(torch::Tensor x, torch::Tensor mean, torch::Tensor cov) {
    // Calculate the Mahalanobis Distance
    auto md = mahalanobis_distance(x, mean, cov);
    auto log_prob = -0.5 * (md + x.size(-1) * torch::log({2 * torch::tensor(3.14159265358979)}) \
                    + torch::logdet(cov.expand({x.size(0), -1, -1, -1})));
    return torch::nan_to_num(log_prob);
}

struct Act_Net : torch::nn::Module {
public:
     Act_Net(){
        int da = n_actions;
        // Construct and register two Linear submodules.
        fc1 = register_module("fc1", torch::nn::Linear(385, 385));
        fc2 = register_module("fc2", torch::nn::Linear(385, 98));
        fc7 = register_module("fc7", torch::nn::Linear(98, 98));
        fc8 = register_module("fc8", torch::nn::Linear(98, da));

        cholesky_layer = register_module("cholesky_layer", torch::nn::Linear(98, floor(da * (da + 1)) / 2));
        fc1->to(device);
        fc2->to(device);
        fc7->to(device);
        fc8->to(device);
        cholesky_layer->to(device);
    }
    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x, int B) {
        int da = n_actions;
        x = torch::elu(fc1->forward(x));
        // x = torch::dropout(x, /*p=*/0.02, /*train=*/is_training());
        x = torch::elu(fc2->forward(x));
        x = torch::tanh(fc7->forward(x));
        // auto mean_layer = torch::(fc6->forward(x));
        auto mean_layer = torch::nan_to_num(fc8->forward(x));
        auto mean = -0.4 + (0.4 - -0.4) * mean_layer;

        auto cholesky_vector = torch::nan_to_num(cholesky_layer->forward(x));
        auto cholesky_vector_size = cholesky_vector.sizes()[0];
        auto cholesky_subvector_size = cholesky_vector[0].sizes()[0];

        auto trl = torch::tril_indices(da, da, 0).to(device);
        auto cholesky = torch::zeros({B, da, da}).to(device);

        auto cholesky_diag_index_ = torch::arange(da) + 1;
        auto cholesky_diag_index = ((cholesky_diag_index_ * (cholesky_diag_index_ + 1)) / 2) - 1;
        cholesky_diag_index = cholesky_diag_index.toType({torch::kInt64});
        auto cholesky_diag_index_flat = cholesky_diag_index.flatten();

        cholesky_vector.index_put_({Slice(None, None, None), cholesky_diag_index}, \
                    torch::softplus(cholesky_vector.index({Slice(None, None, None), cholesky_diag_index})));

        cholesky.index_put_({Slice(None, None, None), trl.index({0}), trl.index({1})}, cholesky_vector);

        return {torch::nan_to_num(mean), torch::nan_to_num(cholesky)};
    }

    torch::Tensor action(torch::Tensor state, int B) {
        torch::NoGradGuard no_grad;
        auto [forward_mean, forward_cholesky] = forward(state, B);
        auto forward_action = MultivariateNormal_sampler(1, forward_mean, \
                                                        forward_cholesky).reshape({-1, n_actions});
        return forward_action;
    }

    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr}, fc4{nullptr}, fc5{nullptr}, \
                    fc6{nullptr}, fc7{nullptr}, fc8{nullptr}, cholesky_layer{nullptr};
};

struct Critic_Net : torch::nn::Module {
    torch::Tensor next_state_batch__sampled_action;
    public:
    Critic_Net() {
        // Construct and register two Linear submodules.
        lin1 = register_module("lin1", torch::nn::Linear(402, 402));
        lin2 = register_module("lin2", torch::nn::Linear(402, 402));
        lin8 = register_module("lin8", torch::nn::Linear(402, 1));
        lin1->to(device);
        lin2->to(device);
        lin8->to(device);
    }
    torch::Tensor forward(torch::Tensor next_state_batch__sampled_action) {
        auto h = next_state_batch__sampled_action;
        h = torch::elu(lin1->forward(h));
        // h = torch::dropout(h, /*p=*/0.02, /*train=*/is_training());
        h = torch::tanh(lin2->forward(h));

        h = lin8->forward(h);
        return torch::nan_to_num(h);
    }

    torch::nn::Linear lin1{nullptr}, lin2{nullptr}, lin8{nullptr};
};

void set_velocity_servo(const mjModel* m, int actuator_no, double kv)
{
  m->actuator_gainprm[10*actuator_no+0] = kv;
  m->actuator_biasprm[10*actuator_no+2] = -kv;
}

auto options = torch::TensorOptions().dtype(torch::kFloat32);
void sim_store_multi_episodes(int id, torch::Tensor* out) {
    torch::Tensor qpos_T = torch::from_blob(d[id]->qpos, {28}, options);
    torch::Tensor qvel_T = torch::from_blob(d[id]->qvel, {28}, options);
    torch::Tensor cinert_T = torch::from_blob(d[id]->cinert, {140}, options);
    torch::Tensor cvel_T = torch::from_blob(d[id]->cvel, {84}, options);
    torch::Tensor qfrc_actuator_T = torch::from_blob(d[id]->qfrc_actuator, {21}, options);
    torch::Tensor cfrc_ext_T = torch::from_blob(d[id]->cfrc_ext, {84}, options);

    torch::Tensor all_obs_T = torch::cat({qpos_T, qvel_T, cinert_T, cvel_T, qfrc_actuator_T, cfrc_ext_T}, 0);

    for (int z=0; z<3; z++) {
        mj_step(m, d[id]);
    }
    *out = all_obs_T.unsqueeze(0);
}

torch::Tensor all_episode_obs() {
    std::thread ths[episode_and_training_batch];
    std::vector<torch::Tensor> results(episode_and_training_batch);
    for (int id=0; id<episode_and_training_batch; id++) {
        ths[id] = std::thread(sim_store_multi_episodes, id, &results[id]);
    }
    for (int id=0; id<episode_and_training_batch; id++) {
        ths[id].join();
    }
    auto result2d = torch::cat(results);
    return result2d.to(device);
}

// Model in 2nd position copied to model in 1st position
void loadstatedict(torch::nn::Module& model, torch::nn::Module& target_model) {
torch::autograd::GradMode::set_enabled(false);  // make parameters copying possible
auto new_params = target_model.named_parameters(); // implement this
auto params = model.named_parameters(true /*recurse*/);
auto buffers = model.named_buffers(true /*recurse*/);
for (auto& val : new_params) {
    auto name = val.key();
    auto* t = params.find(name);
    if (t != nullptr) {
        t->copy_(val.value());
    } else {
        t = buffers.find(name);
        if (t != nullptr) {
            t->copy_(val.value());
            }
        }
    }
    torch::autograd::GradMode::set_enabled(true);
}

auto actor = Act_Net();
auto target_actor = Act_Net();
auto critic = Critic_Net();
auto target_critic = Critic_Net();

auto ε_dual = 0.05; // dual_constraint
auto ε_kl_μ = 0.01; // kl_mean_constraint
auto ε_kl_Σ = 0.0001; // kl_var_constraint
auto ε_kl = 0.001; // kl_constraint

torch::Tensor α_μ_scale = torch::tensor({1.0}).to(device); // alpha_mean_scale
torch::Tensor α_Σ_scale = torch::tensor({100.0}).to(device); // alpha_var_scale
auto α_μ_max = torch::tensor({0.1}).to(device); // alpha_mean_max
auto α_Σ_max = torch::tensor({10.0}).to(device); // alpha_var_max
auto max_loss = torch::tensor({120.0}).to(device); // alpha_var_max

auto α_μ = torch::tensor({0.0}).to(device);  // lagrangian multiplier for continuous action space in the M-step
auto α_Σ = torch::tensor({0.0}).to(device);  // lagrangian multiplier for continuous action space in the M-step

auto γ = 0.999;
auto l1_loss = torch::smooth_l1_loss;
auto actor_optimizer = torch::optim::Adam(actor.parameters(),  3e-6);
auto critic_optimizer = torch::optim::Adam(critic.parameters(),  3e-6);

// mouse interaction
bool button_left = false;
bool button_middle = false;
bool button_right =  false;
double lastx = 0;
double lasty = 0;

// holders of one step history of time and position to calculate dertivatives
mjtNum position_history = 0;
mjtNum previous_time = 0;

// controller related variables
float_t ctrl_update_freq = 100;
mjtNum last_update = 0.0;
mjtNum ctrl;

// keyboard callback
void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods)
{
    // backspace: reset simulation
    if( act==GLFW_PRESS && key==GLFW_KEY_BACKSPACE )
    {
        mj_resetData(m, d[0]);
        mj_forward(m, d[0]);
    }
}

// mouse button callback
void mouse_button(GLFWwindow* window, int button, int act, int mods)
{
    // update button state
    button_left =   (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS);
    button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE)==GLFW_PRESS);
    button_right =  (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT)==GLFW_PRESS);

    // update mouse position
    glfwGetCursorPos(window, &lastx, &lasty);
}

// mouse move callback
void mouse_move(GLFWwindow* window, double xpos, double ypos)
{
    // no buttons down: nothing to do
    if( !button_left && !button_middle && !button_right )
        return;

    // compute mouse displacement, save
    double dx = xpos - lastx;
    double dy = ypos - lasty;
    lastx = xpos;
    lasty = ypos;

    // get current window size
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    // get shift key state
    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)==GLFW_PRESS ||
                      glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT)==GLFW_PRESS);

    // determine action based on mouse button
    mjtMouse action;
    if( button_right )
        action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    else if( button_left )
        action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    else
        action = mjMOUSE_ZOOM;

    // move camera
    mjv_moveCamera(m, action, dx/height, dy/height, &scn, &cam);
}

// scroll callback
void scroll(GLFWwindow* window, double xoffset, double yoffset)
{
    // emulate vertical mouse motion = 5% of window height
    mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05*yoffset, &scn, &cam);
}

torch::Tensor mean_loss_q;
torch::Tensor mean_est_q;

torch::Tensor state_array;
torch::Tensor action_array;
auto reward_array = torch::tensor({}).to(device);
auto done_array = torch::tensor({}).to(device);
auto done_T_F = 0.0;
torch::Tensor next_state_array;
int pre_steps;

std::vector<torch::Tensor> state_vector;
std::vector<torch::Tensor> action_vector;
std::vector<torch::Tensor> reward_vector;
std::vector<torch::Tensor> done_vector;
std::vector<torch::Tensor> next_state_vector;

std::vector<std::vector<torch::Tensor>> state_vector_of_vectors;
std::vector<std::vector<torch::Tensor>> action_vector_of_vectors;
std::vector<std::vector<torch::Tensor>> reward_vector_of_vectors;
std::vector<std::vector<torch::Tensor>> done_vector_of_vectors;
std::vector<std::vector<torch::Tensor>> next_state_vector_of_vectors;

void __sample_trajectory_buffer (int number_of_episodes, int max_steps_in_episode, \
        bool update_buffer, int less_rand, int iteration_number) {

    std::uniform_int_distribution<std::mt19937::result_type> do_nothing_steps(0, 20); // steps before any action taken

    if ((update_buffer==false) && (iteration_number==0)) {
        state_vector_of_vectors.clear();
        action_vector_of_vectors.clear();
        reward_vector_of_vectors.clear();
        done_vector_of_vectors.clear();
        next_state_vector_of_vectors.clear();
    }

    // For loop over. Number of Episodes
    for (int i=0; i<number_of_episodes; ++i) {
        std::cout << " Running episode " << i;

        int sample_episode_maxstep = max_steps_in_episode;
        pre_steps = do_nothing_steps(rng);

        //GET OBS
        state_array = all_episode_obs();
        while (sample_episode_maxstep > 0) {
            sample_episode_maxstep--;

            if (pre_steps > 0) {
            // if (100 < 0) {
                pre_steps--;
                action_array = torch::zeros({episode_and_training_batch, n_actions}).squeeze(0).to(device);
                reward_array = torch::zeros({episode_and_training_batch, 1}).to(device);
            } else {
                if (randk(rng) > less_rand) {
                // if (100 < 0) {
                    action_array = (torch::rand({episode_and_training_batch, n_actions \
                                                }).squeeze(0).to(device) * 2) - 1;
                } else {
                    action_array = actor.action(state_array, episode_and_training_batch);
                }
                // // TAKE ACTIONS
                for (int j=0; j<episode_and_training_batch; ++j) {
                    for (int k=0; k<m->nu; ++k) {
                        // auto kv = action_array[j].index({k+21}).item().toDouble();
                        // set_velocity_servo(m, k, kv);
                        // d[index] is one of many simulations running
                        // d[index]->ctrl[index] is one of many actuators 
                        d[j]->ctrl[k] = action_array[j].index({k}).item().toDouble();
                    }
                    // SET THE REWARD ~ (Y disance of head from the ground)
                    reward_array = torch::cat({torch::tensor({d[j]->sensordata[2]}).to(device), reward_array}, 0);
                    // if ((((d[j]->qpos[1]*-1)+1.57) < 0.6) || (sample_episode_maxstep == 0)) {
                    // if (((d[j]->qpos[1]*-1)+1.57) < 0.6) {
                    //     done_T_F = 1.0;
                    // } else {
                    //     done_T_F = 0.0;
                    // }
                    done_array = torch::cat({torch::tensor({done_T_F}).to(device), done_array}, 0);
                    // done_array = torch::cat({torch::tensor({(d[j]->qpos[1]*-1)+1.57}).to(device), done_array}, 0);

                }
                reward_array = reward_array.unsqueeze(-1);
                done_array = done_array.unsqueeze(-1);
            }

            //GET OBS AFTER ACTIONS TAKEN
            auto next_state_array = all_episode_obs();

            state_vector.push_back(state_array);
            action_vector.push_back(action_array);
            reward_vector.push_back(reward_array);
            done_vector.push_back(done_array);
            next_state_vector.push_back(next_state_array);

            state_array = next_state_array;
            reward_array = torch::tensor({}).to(device);
            done_array = torch::tensor({}).to(device);
        }

        for (int index=0; index<episode_and_training_batch; index++) {
            mj_resetData(m, d[index]);
        }

        if (update_buffer==true) {
            state_vector_of_vectors.erase(state_vector_of_vectors.begin());
            action_vector_of_vectors.erase(action_vector_of_vectors.begin());
            reward_vector_of_vectors.erase(reward_vector_of_vectors.begin());
            done_vector_of_vectors.erase(done_vector_of_vectors.begin());
            next_state_vector_of_vectors.erase(next_state_vector_of_vectors.begin());
        }

        state_vector_of_vectors.push_back(state_vector);
        action_vector_of_vectors.push_back(action_vector);
        reward_vector_of_vectors.push_back(reward_vector);
        done_vector_of_vectors.push_back(done_vector);
        next_state_vector_of_vectors.push_back(next_state_vector);

        state_vector.clear();
        action_vector.clear();
        reward_vector.clear();
        done_vector.clear();
        next_state_vector.clear();
    }
}

std::vector<std::vector<float>> reward_means(std::vector<std::vector<float>> the_array) {
    float m_sum;
    float m_mean;
    float n;
    std::vector<float> mean_vector;
    std::vector<float> sum_vector;
    std::vector<std::vector<float>> return_vector;
    for (float i=0; i<the_array.size(); i++) {
        n = 0;
        for (float x=0; x<the_array[i].size(); x++) {
            n += the_array[i][x];
        }
        auto mean = n / the_array[i].size();
        m_sum += n;
        m_mean += mean;
        mean_vector.insert(mean_vector.begin(), m_mean / the_array.size());
        sum_vector.insert(sum_vector.end(), m_sum / the_array.size());
    }
    return_vector.insert(return_vector.begin(), mean_vector);
    return_vector.insert(return_vector.begin(), sum_vector);
    return return_vector;
}
// Policy Evaluation
std::tuple<torch::Tensor, torch::Tensor> __update_critic_td_1(
                                torch::Tensor reward_batch,
                                torch::Tensor next_state_batch
                                ) {
    torch::NoGradGuard no_grad;
        auto [π_μ, π_A] = target_actor.forward(next_state_batch, B_size);

        auto sampled_actions = MultivariateNormal_sampler(1, π_μ, π_A).reshape({-1, n_actions});
        auto next_state_and_sampled_actions = torch::cat({sampled_actions, \ 
                                    next_state_batch.to(device)}, 1);
        auto expected_next_q = target_critic.forward(next_state_and_sampled_actions);

        auto y = torch::nan_to_num(reward_batch + γ * expected_next_q);
        return {torch::nan_to_num(y), torch::nan_to_num(sampled_actions)};
}

torch::Tensor __update_critic_td_2(torch::Tensor state_batch,
                                torch::Tensor action_batch,
                                torch::Tensor y
                                ) {
    critic_optimizer.zero_grad();
    auto state_and_act = torch::cat({action_batch, state_batch}, 1);
    auto t = critic.forward(state_and_act.squeeze(-1)).to(device);
    auto loss = torch::smooth_l1_loss(y, t);
    loss.backward();
    // for (auto& param : critic.parameters()) {
    //     torch::nn::utils::clip_grad_norm_(param, 0.1);
    // }
    // torch::nn::utils::clip_grad_norm_(critic.parameters(), 0.1);
    critic_optimizer.step();
    return torch::nan_to_num(loss);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> e_step(torch::Tensor state_batch) {
    torch::NoGradGuard no_grad;
        auto [b_μ, b_A] = target_actor.forward(state_batch, B_size);
        auto sampled_actions = MultivariateNormal_sampler(20, b_μ, b_A);
        auto expanded_state_batch = state_batch.expand({20, -1, -1}).reshape({-1, n_obs_numbers});
        auto next_state_and_sampled_actions = torch::cat({sampled_actions.reshape({-1, n_actions}), \
                                                        expanded_state_batch}, -1);
        auto target_q = target_critic.forward(next_state_and_sampled_actions).reshape({20, B_size});
        return {torch::nan_to_num(target_q), torch::nan_to_num(sampled_actions), \
                torch::nan_to_num(b_μ), torch::nan_to_num(b_A)};
}

struct read_write_tensors {
    std::vector<std::vector<torch::Tensor>> vec_of_vec_of_tensor;
    std::vector<torch::Tensor> vec_of_tensor;
    std::string folder_name;
    int episode_length;
    torch::Tensor er;
    public:
    void write_tensors(std::vector<std::vector<torch::Tensor>> vec_of_vec_of_tensor, \
        std::string folder_name, int episode_length) {
        for (int w = 0; w<episode_and_training_batch; w++) {
            auto hoo = fmt::format("{0:0{1}}", w, 3);
            auto the_path = "upsidedown_pendulum/" + folder_name + "/" + hoo;
            for (int oa = 0; oa<episode_length; oa++) {
                auto each_one = fmt::format("{0:0{1}}.pt", oa, 4);
                auto path_w_file = the_path + "/" + each_one;
                torch::save(vec_of_vec_of_tensor[w][oa], path_w_file);
            }
        }
    }
    std::vector<std::vector<torch::Tensor>> read_tensors(std::string folder_name, \
        int episode_length) {
        std::vector<std::vector<torch::Tensor>> vec_of_vec_of_tensor;
        std::vector<torch::Tensor> vec_of_tensor;
        for (int w = 0; w<episode_and_training_batch; w++) {
            vec_of_tensor.clear();
            auto hoo = fmt::format("{0:0{1}}", w, 3);
            auto the_path = "upsidedown_pendulum/" + folder_name + "/" + hoo;
            for (int oa = 0; oa<episode_length; oa++) {
                er = torch::tensor({});
                auto each_one = fmt::format("{0:0{1}}.pt", oa, 4);
                auto path_w_file = the_path + "/" + each_one;
                torch::load(er, path_w_file);
                vec_of_tensor.push_back(er.to(device));
            }
            vec_of_vec_of_tensor.push_back(vec_of_tensor);
        }
        return vec_of_vec_of_tensor;
    }
};

struct read_write_integers {
    public:
    void write_integers(std::vector<std::vector<int>> int_vec_of_vec, \
    std::string folder_name) {
        for (int z = 0; z<int_vec_of_vec.size(); z++) {
            std::ofstream file_in;
            auto hoo = fmt::format("{0:0{1}}", z, 3);
            auto the_path = "upsidedown_pendulum/" + folder_name + "/" + hoo + "_integers.txt";
            file_in.open (the_path);
            for (int i = 0; i<int_vec_of_vec[0].size(); i++) {
                file_in << int_vec_of_vec[z][i];
                file_in << std::endl;
            }
            file_in.close();
        }
    }
    std::vector<std::vector<int>> read_integers(std::string folder_name) {
        std::vector<std::vector<int>> int_vec_of_vec;
        std::vector<int> int_vec;
        for (int z = 0; z<episode_and_training_batch; z++) {
            int_vec.clear();
            // // Read ints and store to vector
            std::ifstream myReadFile;
            auto hoo = fmt::format("{0:0{1}}", z, 3);
            auto the_path = "upsidedown_pendulum/" + folder_name + "/" + hoo + "_integers.txt";
            myReadFile.open(the_path);
            int n;                     // ****
            while (myReadFile >> n) {        // ****
                int_vec.push_back(n);      // ****
            }
            myReadFile.close();
            int_vec_of_vec.push_back(int_vec);
        }
    return int_vec_of_vec;
    }
};

struct read_write_floats {
    public:
    void write_floats(std::vector<float> float_vec_of_vec) {
        // for (int z = 0; z<float_vec_of_vec.size(); z++) {
            std::ofstream file_in;
            auto hoo = fmt::format("{0:0{1}}", 100, 3);
            auto the_path = "i" + hoo + "_floats.txt";
            file_in.open (the_path);
            for (int i = 0; i<float_vec_of_vec.size(); i++) {
                file_in << float_vec_of_vec[i];
                file_in << std::endl;
            }
            file_in.close();
        // }
    }

};

auto read_write_tensors_ = read_write_tensors();
auto read_write_integers_ = read_write_integers();
auto read_write_floats_ = read_write_floats();

std::vector<float> reward_mean_vector_of_vectors;
float read_me;

void train(bool load_obs) {
    auto η = torch::rand(1, torch::requires_grad()).to(device);
    auto eta = η.item();
    torch::Tensor dual_function;

    // torch::serialize::InputArchive actor_archive;
    // std::string actor_model_path = "actor_humanoidstand.pt";
    // actor_archive.load_from(actor_model_path);
    // actor.load(actor_archive);

    // torch::serialize::InputArchive critic_archive;
    // std::string critic_model_path = "critic_humanoidstand.pt";
    // critic_archive.load_from(critic_model_path);
    // critic.load(critic_archive);

    loadstatedict(target_actor, actor);
    loadstatedict(target_critic, critic);

    torch::Tensor all_step_obs;
    torch::Tensor loss_p;
    torch::Tensor loss_l;
    torch::Tensor output;
    torch::Tensor π_μ;
    torch::Tensor π_A;
    torch::Tensor qij;
    torch::Tensor mean_last_episodes_rewards = torch::tensor({}).to(device);

    torch::Tensor full_state_batch = torch::tensor({}).to(device);
    torch::Tensor full_next_state_batch = torch::tensor({}).to(device);
    torch::Tensor full_action_batch = torch::tensor({}).to(device);
    torch::Tensor full_reward_batch = torch::tensor({}).to(device);
    torch::Tensor one_state_batch;
    torch::Tensor one_next_state_batch;
    torch::Tensor one_action_batch;
    torch::Tensor one_reward_batch;
    int mean_episodes_start = episode_and_training_batch - 40;

    int x;
    int z;
    int za;

    auto min_optimizer = torch::optim::SGD({torch::rand({1})}, 0.1);
    int training_steps = 1000000000;
    int iteration_num = episode_and_training_batch;
    int episode_length = 350;
    // int roll_buffer = iteration_num * episode_length;
    int roll_buffer = B_size - 1;
    int iteration_num_idx = iteration_num - 1;
    int see_episode_reward_slice = iteration_num_idx - 10;
    int episode_length_idx = episode_length - 1;
    int target_update_frequency = 350;
    auto update_modulus = target_update_frequency - 1;

    std::uniform_int_distribution<std::mt19937::result_type> dist_episodes(0, iteration_num_idx); // distribution in range [1, 6]
    std::uniform_int_distribution<std::mt19937::result_type> dist_episode_length(0, episode_length_idx); // distribution in range [1, 6]
    std::uniform_int_distribution<std::mt19937::result_type> see_episode_reward(see_episode_reward_slice, \
                                                                                iteration_num_idx);
    if (load_obs==false) {
        __sample_trajectory_buffer(iteration_num, episode_length, false, 0, 0);
        read_write_tensors_.write_tensors(state_vector_of_vectors, "state", episode_length);
        read_write_tensors_.write_tensors(next_state_vector_of_vectors, "next_state", episode_length);
        read_write_tensors_.write_tensors(action_vector_of_vectors, "action", episode_length);
        read_write_tensors_.write_tensors(done_vector_of_vectors, "done", episode_length);
        read_write_tensors_.write_tensors(reward_vector_of_vectors, "reward", episode_length);
    } else {
        std::cout << "Reading states, actions, and rewards from disk... \n";
        state_vector_of_vectors = read_write_tensors_.read_tensors("state", episode_length);
        next_state_vector_of_vectors = read_write_tensors_.read_tensors("next_state", episode_length);
        action_vector_of_vectors = read_write_tensors_.read_tensors("action", episode_length);
        done_vector_of_vectors = read_write_tensors_.read_tensors("done", episode_length);
        reward_vector_of_vectors = read_write_tensors_.read_tensors("reward", episode_length);

    }

    mean_last_episodes_rewards = torch::tensor({}).to(device);
    for (int e=0; e<episode_and_training_batch; e++) {
        reward_vector_of_vectors[iteration_num_idx][e];
        mean_last_episodes_rewards = torch::cat({( \
                    torch::mean(reward_vector_of_vectors[iteration_num_idx][e]).unsqueeze(0)), \ 
                    mean_last_episodes_rewards}, 0);
    }

    for (int z_=0; z_<training_steps; z_++) {
        // z = dist_episodes(rng);
        // x = dist_episode_length(rng);
        // full_state_batch = state_vector_of_vectors[z][x];
        // full_next_state_batch = next_state_vector_of_vectors[z][x];
        // full_action_batch = action_vector_of_vectors[z][x];
        // full_reward_batch = reward_vector_of_vectors[z][x];
        for (int zx=0; zx<batch_multiple; zx++) {
            z = dist_episodes(rng);
            x = dist_episode_length(rng);
            one_state_batch = state_vector_of_vectors[z][x];
            one_next_state_batch = next_state_vector_of_vectors[z][x];
            one_action_batch = action_vector_of_vectors[z][x];
            one_reward_batch = reward_vector_of_vectors[z][x];

            full_state_batch = torch::cat({(torch::nan_to_num(one_state_batch)), \ 
                                                full_state_batch}, 0);
            full_next_state_batch = torch::cat({(torch::nan_to_num(one_next_state_batch)), \ 
                                                full_next_state_batch}, 0);
            full_action_batch = torch::cat({(torch::nan_to_num(one_action_batch)), \ 
                                                full_action_batch}, 0);
            full_reward_batch = torch::cat({(torch::nan_to_num(one_reward_batch)), \ 
                                                full_reward_batch}, 0);
        }
        // Policy Evaluation
        auto [q, td_1_sampled_actions] = __update_critic_td_1(full_reward_batch,
                                                            torch::nan_to_num(full_next_state_batch));

        auto loss_q = __update_critic_td_2(torch::nan_to_num(full_state_batch), nan_to_num(full_action_batch), q);

        auto [target_q_T, e_sampled_actions, b_μ, b_A] = e_step(torch::nan_to_num(full_state_batch));
        auto [max_q, _] = target_q_T.max(1);

        η = η.clone().detach().requires_grad_();

        auto min_optimizer = torch::optim::SGD({η}, 0.1);
        for (int c=0; c<150; c++) {
            min_optimizer.zero_grad();
            dual_function = η * ε_dual + torch::mean(max_q) + η \
                * torch::mean(torch::log(torch::mean(torch::exp((target_q_T - max_q.unsqueeze(1)) / η))));
            dual_function.backward();
            // if (c % 20 == 0) {
            //     std::cout << η.item() << " " << c << " η.item() \n ";
            // }
            min_optimizer.step();
        }
        eta = η.item();

        qij = torch::softmax(target_q_T / η, 0);
        qij = qij.clone().detach().requires_grad_();
        η = torch::rand(1, torch::requires_grad()).to(device);

        for (int ae=0; ae<5; ae++) {
            auto [μ, A] = actor.forward(torch::nan_to_num(full_state_batch), B_size);

            // loss_p = torch::mean(qij * (log_prob(μ.expand({20, -1, -1}), b_A.expand({20, -1, -1, -1}), \
            //                             e_sampled_actions)
            //                             + \
            //                             log_prob(b_μ.expand({20, -1, -1}), A.expand({20, -1, -1, -1}), \
            //                             e_sampled_actions)));

            auto lp1 = log_prob_mvn(e_sampled_actions, μ, b_A);
            auto lp2 = log_prob_mvn(e_sampled_actions, b_μ, A);

            loss_p = torch::mean(qij * (lp1 + lp2));

            auto [kl_μ, kl_Σ, Σi_det, Σ_det] = gaussian_kl(b_μ, μ, b_A, A);

            α_μ -= α_μ_scale * (ε_kl_μ - kl_μ).detach().item();
            α_Σ -= α_Σ_scale * (ε_kl_Σ - kl_Σ).detach().item();

            α_μ = torch::clip(torch::tensor({0.0}).to(device), α_μ, α_μ_max);
            α_Σ = torch::clip(torch::tensor({0.0}).to(device), α_Σ, α_Σ_max);

            actor_optimizer.zero_grad();

            loss_l = -(loss_p + α_μ * (ε_kl_μ - kl_μ) + α_Σ * (ε_kl_Σ - kl_Σ));

            if (ae % 5 == 0) {
                std::cout << "\n" << loss_q.item() << " <-- __update_critic_td_2 loss";
                std::cout << "\n" << loss_l.item() << " <-loss_l " << loss_p.item() << " <-loss_p " << \
                        z_ << " <-- current step \n";
                std::cout << "\n" << α_μ.item() << "<-- α_μ " << α_Σ.item() << "<-- α_Σ " << z_ << "\n";
                std::cout << "\n" << kl_μ.item() << "<-- kl_μ " << kl_Σ.item() << "<-- kl_Σ " << z_ << "\n";
                std::cout << "\n" << torch::mean(e_sampled_actions).item() << " e_sampled_actions \n";
                std::cout << torch::mean(qij).item() << " qij \n";
                std::cout << torch::mean(target_q_T).item() << " target_q_T \n";

                z = dist_episodes(rng);
                za = see_episode_reward(rng);
                x = dist_episode_length(rng);
                std::cout << torch::mean(reward_vector_of_vectors[z][x]).item() << " rand_mean of the whole thing \n";
                std::cout << torch::mean(reward_vector_of_vectors[za][x]).item() << " rand_mean of the last ten \n";
                std::cout << torch::mean(reward_vector_of_vectors[iteration_num_idx][x]).item() << " mean of the last one \n";
                std::cout << torch::mean(mean_last_episodes_rewards).item() << " mean_last_episodes_rewards \n";

                std::cout << torch::mean(μ).item().toBool() << " μ.item().toBool() \n";

                std::cout << torch::mean(b_μ).item() << " b_μ \n";
                std::cout << torch::mean(μ).item() << " μ \n";
                std::cout << torch::mean(b_A).item() << " b_A \n";
                std::cout << torch::mean(A).item() << " A \n";

                std::cout << eta << " eta \n";
            }

            loss_l.backward();
            torch::nn::utils::clip_grad_norm_(actor.parameters(), 0.1);
            actor_optimizer.step();
        }

        // Mod is stored episodes * steps in episode / batch_multiple * steps in episode
        // if ((z_ % B_size == roll_buffer) || (z_ == 0)) {
        if ((z_ % 125 == 119) && (z_ > 500)) {
            std::cout << "Adding new episode to the stack. \n";
            __sample_trajectory_buffer(4, episode_length, true, z_, z_);

            if (z_ % 10000 == 9999) {
                read_write_tensors_.write_tensors(state_vector_of_vectors, "state", episode_length);
                std::cout << "how long did this 1st one take? \n";
                read_write_tensors_.write_tensors(next_state_vector_of_vectors, "next_state", episode_length);
                std::cout << "how long did this 2st one take? \n";
                read_write_tensors_.write_tensors(action_vector_of_vectors, "action", episode_length);
                std::cout << "how long did this 3st one take? \n";
                read_write_tensors_.write_tensors(done_vector_of_vectors, "done", episode_length);
                std::cout << "how long did this 4st one take? \n";
                read_write_tensors_.write_tensors(reward_vector_of_vectors, "reward", episode_length);
                std::cout << "how long did this 5st one take? \n";
            }

            mean_last_episodes_rewards = torch::tensor({}).to(device);
            for (int e=0; e<episode_and_training_batch; e++) {
                reward_vector_of_vectors[iteration_num_idx][e];
                mean_last_episodes_rewards = torch::cat({( \
                            torch::mean(reward_vector_of_vectors[iteration_num_idx][e]).unsqueeze(0)), \ 
                            mean_last_episodes_rewards}, 0);
            }

            read_me = torch::mean(mean_last_episodes_rewards).item<float>();
            reward_mean_vector_of_vectors.push_back(read_me);
            read_write_floats_.write_floats(reward_mean_vector_of_vectors);

            std::cout << " save models begin \n";
            std::string actor_model_path = "actor_humanoidstand.pt";
            torch::serialize::OutputArchive actor_output_archive;
            actor.save(actor_output_archive);
            actor_output_archive.save_to(actor_model_path);

            std::string critic_model_path = "critic_humanoidstand.pt";
            torch::serialize::OutputArchive critic_output_archive;
            critic.save(critic_output_archive);
            critic_output_archive.save_to(critic_model_path);
            std::cout << " save models end \n";
        }

        if (z_ % target_update_frequency == update_modulus) {
            for (int xox = 0; xox<40; xox++) {
                std::cout << "targets updated \n \n";
                std::cout << "targets updated \n \n";
            }

            loadstatedict(target_actor, actor);
            loadstatedict(target_critic, critic);

        }
        full_state_batch = torch::tensor({}).to(device);
        full_next_state_batch = torch::tensor({}).to(device);
        full_action_batch = torch::tensor({}).to(device);
        full_reward_batch = torch::tensor({}).to(device);
    }
};


// create OpenGL context/window
void initOpenGL(void) {
  // init GLFW
  if (!glfwInit()) {
    mju_error("Could not initialize GLFW");
  }
}

// close OpenGL context/window
void closeOpenGL(void) {
  // terminate GLFW (crashes with Linux NVidia drivers)
  #if defined(__APPLE__) || defined(_WIN32)
    glfwTerminate();
  #endif
}


int main (int argc, const char** argv) {

    mj_activate("mjkey.txt");
    // load and compile model
    char error[1000] = "Could not load binary model";

    // check command-line arguments
    if( argc<2 )
        m = mj_loadXML(filename, 0, error, 1000);
    else
        if( strlen(argv[1])>4 && !strcmp(argv[1]+strlen(argv[1])-4, ".mjb") )
            m = mj_loadModel(argv[1], 0);
        else
            m = mj_loadXML(argv[1], 0, error, 1000);
    if( !m )
        mju_error_s("Load model error: %s", error);

    // make per-thread data
    int testkey = mj_name2id(m, mjOBJ_KEY, "test");
    for (int id=0; id<episode_and_training_batch; id++) {
        d[id] = mj_makeData(m);
        if (!d[id]) {
        return finish("Could not allocate mjData", m);
        }

        // init to keyframe "test" if present
        if (testkey>=0) {
        mju_copy(d[id]->qpos, m->key_qpos + testkey*m->nq, m->nq);
        mju_copy(d[id]->qvel, m->key_qvel + testkey*m->nv, m->nv);
        mju_copy(d[id]->act,  m->key_act  + testkey*m->na, m->na);
        }
    }

    train(1);

}