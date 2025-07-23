import torch
import os
import numpy as np
import h5py
from utils.model_utils import RUNCONFIGS #, get_dataset_name
import copy
import torch.nn.functional as F
import torch.nn as nn
from utils.model_utils import get_log_path, METRICS

class Server:
    def __init__(self, args, model, seed, logging):
        self.logging = logging
        # Set up the main attributes
        self.dataset = args.dataset
        self.num_glob_iters = args.num_glob_iters
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.total_train_samples = 0
        self.K = args.K
        self.model = copy.deepcopy(model[0])
        self.model_name = model[1]
        self.users = []
        self.selected_users = []
        self.num_users = args.num_users
        self.beta = args.beta
        self.lamda = args.lamda
        self.algorithm = args.algorithm
        self.personalized = 'pFed' in self.algorithm
        self.mode='partial' if 'partial' in self.algorithm.lower() else 'all'
        self.seed = seed
        self.deviations = {}
        self.metrics = {key:[] for key in METRICS}
        self.timestamp = None
        self.save_path = args.result_path
        os.system("mkdir -p {}".format(self.save_path))

        # 自己新增
        self.device = args.device
        self.best_accu = 0
        self.best_loss = 1e9
        self.best_iter = -1

    def send_parameters(self, mode='all', beta=1, selected=False):
        users = self.users
        if selected:
            assert (self.selected_users is not None and len(self.selected_users) > 0)
            users = self.selected_users
        for user in users:
            if mode == 'all': # share only subset of parameters
                user.set_parameters(self.model,beta=beta)
            else: # share all parameters
                user.set_shared_parameters(self.model,mode=mode)

    def aggregate_parameters(self):
        ## TODO
        '''
        Weighted sum all the selected users' model parameters by number of samples

        Args: None
        Return: None

        Hints:
            1. Use self.selected_users, user.train_samples.
            2. Replace the global model (self.model) with the aggregated model.
        '''
        # 獲取全域模型的 state_dict ，將其用作模板，並初始化 aggregated_params ，使其具有與全域模型相同的結構，且設初始值為零
        global_model_params = self.model.state_dict()
        aggregated_params = {key: torch.zeros_like(param.data) for key, param in global_model_params.items()}

        # 初始化本輪選中使用者的總訓練樣本數
        total_train_samples_in_round = 0

        # 遍歷每一位被選中的使用者
        for user in self.selected_users:
            # 如果使用者的訓練樣本數為 0，則記錄偵錯訊息並跳過此使用者。
            if user.train_samples == 0:
                self.logging.debug(f"User {user.id} has 0 training samples. Skipping for aggregation.") #
                continue

            # 累加本輪的總訓練樣本數
            total_train_samples_in_round += user.train_samples
            # 獲取使用者本地模型的參數 (state_dict)
            local_model_params = user.model.state_dict()

            # 遍歷全域模型的每一個參數層 (例如，權重、偏差)
            for key in global_model_params.keys():
                if global_model_params[key].dtype == torch.float32:
                    # 累加加權後的參數：本地模型參數 * 該使用者的訓練樣本數
                    aggregated_params[key] += local_model_params[key] * user.train_samples

        # 檢查本輪是否有實際處理的樣本
        if total_train_samples_in_round == 0:
            self.logging.warning("Total training samples for selected users in this round is 0. Skipping model aggregation.")
            return

        # 計算加權平均：將累加後的參數除以本輪的總訓練樣本數
        for key in aggregated_params.keys():
            if aggregated_params[key].dtype == torch.float32:
              aggregated_params[key] /= total_train_samples_in_round

        # 將新聚合得到的參數載入到全域模型中
        self.model.load_state_dict(aggregated_params)

    def save_model(self):
        model_path = os.path.join(self.save_path, self.algorithm, "models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "best_server" + ".pt"))


    def load_model(self):
        model_path = os.path.join(self.save_path, "models", self.dataset, "server" + ".pt")
        assert (os.path.exists(model_path))
        self.model = torch.load(model_path)

    def model_exists(self):
        return os.path.exists(os.path.join("models", self.dataset, "server" + ".pt"))

    def select_users(self, round, num_users):
        ## TODO
        '''
        Randomly select {num_users} users from all users
        Args:
            round: current round
            num_users: number of users to select
        Return:
            List of selected clients objects

        Hints:
            1. Default 10 users to select, you can modify the args {--num_users} to change this hyper-parameter
            2. Note that {num_users} can not be larger than total users (i.e., num_users <= len(self.user))
        '''
        # 如果要選取的使用者數量大於總使用者數
        if num_users > len(self.users):
            # 記錄警告訊息，並將選取數量設為總使用者數(選取所有使用者)
            self.logging.warning(f"Number of users to select ({num_users}) is greater than total available users ({len(self.users)}). Selecting all users.")
            num_users = len(self.users)

        # 如果要選取的使用者數量為 0
        if num_users == 0:
            self.logging.warning("Number of users to select is 0. Returning an empty list.")
            return []
        # 從所有使用者的索引中，無放回地隨機選取 num_users 個索引
        selected_user_indices = np.random.choice(
            len(self.users),  # 可供選擇的總數
            num_users,        # 要選取的數量
            replace=False     # 設定為 False，確保同一使用者不會被選取多次
        )

        # 根據選中的索引，從 self.users 列表中取出對應的使用者物件
        selected_users_list = [self.users[i] for i in selected_user_indices]

        # 回傳被選中的使用者物件列表
        return selected_users_list

    def init_loss_fn(self):
        self.loss= nn.CrossEntropyLoss()# nn.NLLLoss()
        self.ensemble_loss=nn.KLDivLoss(reduction="batchmean")#,log_target=True)
        self.ce_loss = nn.CrossEntropyLoss()


    def save_results(self, args):
        alg = get_log_path(args, args.algorithm, self.seed, args.gen_batch_size)
        with h5py.File("./{}/{}.h5".format(self.save_path, alg), 'w') as hf:
            for key in self.metrics:
                hf.create_dataset(key, data=self.metrics[key])
            hf.close()


    def test(self, selected=False):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        losses = []
        users = self.selected_users if selected else self.users
        for c in users:
            # print(f"client id: {c.id}")
            ct, c_loss, ns = c.test()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(c_loss)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct, losses



    def test_personalized_model(self, selected=True):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        losses = []
        users = self.selected_users if selected else self.users
        for c in users:
            ct, ns, loss = c.test_personalized_model()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(loss)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct, losses

    def evaluate_personalized_model(self, selected=True, save=True):
        stats = self.test_personalized_model(selected=selected)
        test_ids, test_num_samples, test_tot_correct, test_losses = stats[:4]
        glob_acc = np.sum(test_tot_correct)*1.0/np.sum(test_num_samples)
        test_loss = np.sum([x * y for (x, y) in zip(test_num_samples, test_losses)]).item() / np.sum(test_num_samples)
        if save:
            self.metrics['per_acc'].append(glob_acc)
            self.metrics['per_loss'].append(test_loss)
        print("Average Global Accurancy = {:.4f}, Loss = {:.2f}.".format(glob_acc, test_loss))


    def evaluate_ensemble(self, selected=True):
        self.model.eval()
        users = self.selected_users if selected else self.users
        test_acc=0
        loss=0
        for x, y in self.testloaderfull:
            target_logit_output=0
            for user in users:
                # get user logit
                user.model.eval()
                user_result=user.model(x, logit=True)
                target_logit_output+=user_result['logit']
            target_logp=F.log_softmax(target_logit_output, dim=1)
            test_acc+= torch.sum( torch.argmax(target_logp, dim=1) == y ) #(torch.sum().item()
            loss+=self.loss(target_logp, y)
        loss = loss.detach().numpy()
        test_acc = test_acc.detach().numpy() / y.shape[0]
        self.metrics['glob_acc'].append(test_acc)
        self.metrics['glob_loss'].append(loss)
        print("Average Global Accurancy = {:.4f}, Loss = {:.2f}.".format(test_acc, loss))


    def evaluate(self, iter, save=True, selected=False):
        # override evaluate function to log vae-loss.
        test_ids, test_samples, test_accs, test_losses = self.test(selected=selected)
        glob_acc = np.sum(test_accs)*1.0/np.sum(test_samples)
        test_losses = [t.detach().cpu().numpy() for t in test_losses] # Error: RuntimeError: Can't call numpy() on Tensor that requires grad. #15
        glob_loss = np.sum([x * y for (x, y) in zip(test_samples, test_losses)]).item() / np.sum(test_samples)
        if save:
            self.metrics['glob_acc'].append(glob_acc)
            self.metrics['glob_loss'].append(glob_loss)
        # print("Average Global Accurancy = {:.4f}, Loss = {:.2f}.".format(glob_acc, glob_loss))

        if glob_acc > self.best_accu:
            self.best_accu = glob_acc
            self.best_loss = glob_loss
            self.best_iter = iter
            self.save_model()


        self.logging.info("Average Global Accurancy = {:.4f}, Loss = {:.2f}.".format(glob_acc, glob_loss))
        self.logging.info("Best Global Accurancy = {:.4f}, Loss = {:.2f}, Iter = {:}.".format(self.best_accu, self.best_loss, self.best_iter))
