import skopt
import numpy as np
import pandas as pd


class Controller:
    opt = None  # for bayesian optimization
    random_search_space = None # for random search

    def __init__(self, args):
        self.args = args
        # Initiliaze Controller either as a Bayesian Optimizer or as a Random Search Algorithm
        if args.method == 'bayesian_optimization':
            self.init_skopt(args.opt_initial_points)
        elif args.method == 'random':
            self.init_random_search()
        else:
            raise ValueError

    # todo : 여기 5차원의 값이 리턴되는지 확인이 필요 딱 봐선, 4 * 5 차원 같음.
    def init_skopt(self, opt_initial_points):
        # Initialize as scikit-optimize (skopt) Optimizer with a 5-dimensional search space
        # Aligned with skopt ask-tell design pattern (https://geekyisawesome.blogspot.com/2018/07/hyperparameter-tuning-using-scikit.html)
        # Args: opt_initial_points (int): number of random initial points for the optimizer

        self.opt = skopt.Optimizer(
            [
                skopt.space.Categorical(self.args.aug_types, name="A_aug1_type"),
                skopt.space.Real(0.0, 1.0, name="A_aug1_magnitude"),
                skopt.space.Categorical(self.args.aug_types, name="A_aug2_type"),
                skopt.space.Real(0.0, 1.0, name="A_aug2_magnitude"),

                skopt.space.Categorical(self.args.aug_types, name="B_aug1_type"),
                skopt.space.Real(0.0, 1.0, name="B_aug1_magnitude"),
                skopt.space.Categorical(self.args.aug_types, name="B_aug2_type"),
                skopt.space.Real(0.0, 1.0, name="B_aug2_magnitude"),

                skopt.space.Categorical(self.args.aug_types, name="C_aug1_type"),
                skopt.space.Real(0.0, 1.0, name="C_aug1_magnitude"),
                skopt.space.Categorical(self.args.aug_types, name="C_aug2_type"),
                skopt.space.Real(0.0, 1.0, name="C_aug2_magnitude"),

                skopt.space.Categorical(self.args.aug_types, name="D_aug1_type"),
                skopt.space.Real(0.0, 1.0, name="D_aug1_magnitude"),
                skopt.space.Categorical(self.args.aug_types, name="D_aug2_type"),
                skopt.space.Real(0.0, 1.0, name="D_aug2_magnitude"),

                skopt.space.Categorical(self.args.aug_types, name="E_aug1_type"),
                skopt.space.Real(0.0, 1.0, name="E_aug1_magnitude"),
                skopt.space.Categorical(self.args.aug_types, name="E_aug2_type"),
                skopt.space.Real(0.0, 1.0, name="E_aug2_magnitude")
            ],
            n_initial_points=opt_initial_points,
            base_estimator="RF",
            acq_func="EI",
            acq_optimizer="auto",
            random_state=0
        )

    def init_random_search(self):
        self.random_search_space = [
            np.random.choice(self.args.aug_types),
            np.random.rand,
            np.random.choice(self.args.aug_types),
            np.random.rand,
            np.random.rand,
        ]

    def ask(self):
        """Ask controller for the next hyperparameter search.
        If Bayesian Optimizer, samples next hyperparameters by its internal statistic calculations (Random Forest Estimators, Gaussian Processes, etc.). If Random Search, samples randomly
        Based on ask-tell design pattern

        Returns:
            list: list of hyperparameters
        """
        if self.args.method == 'bayesian_optimization':
            return self.opt.ask()
        elif self.args.method == 'random_search':
            return [func() for func in self.random_search_space]

    def tell(self, trial_hyperparams, f_val):
        """
        Tells the controller result of previous tried hyperparameters
        If Bayesian Optimizer, records this results and updates its internal statistical model.
        If Random Search does nothing, since it will not affect future (random) samples.
        :param trial_hyperparams:
        :param f_val:
        :return:
        """
        if self.args.method == 'bayesian_optimization':
            self.args.logger.info(f"previous tried with hyper param {trial_hyperparams} and get the reward {f_val}")
            self.opt.tell(trial_hyperparams, f_val)
        elif self.args.method == 'random_search':
            pass

class Notebook:
    def __init__(self, args):
        self.args = args
        self.df = pd.DataFrame()

    def record(self, trial_no, trial_hyperparams, sample_no, reward, history):
        """Records one complete training of child model

        Args:
            trial_no (int): no of trial (iteration) of training
            trial_hyperparams (list) : list of data augmentation hyperparameters used for training
            sample_no (int): sample no among training with same hyperparameters
            reward (float): reward is basically last n validation accuracy before overfitting
            history (dict): history returned by keras.model.fit()
        """
        train_history, valid_history = history
        df_train_history, df_valid_history = pd.DataFrame(train_history), pd.DataFrame(valid_history)
        df_train_history.columns = ["train_" + c for c in df_train_history.columns]
        df_valid_history.columns = ["valid_" + c for c in df_valid_history.columns]
        df_new = df_train_history.join(df_valid_history)
        df_new['trial_no'] = trial_no
        
        df_new["A_aug1_type"] = trial_hyperparams[0]
        df_new["A_aug1_magnitude"] = trial_hyperparams[1]
        df_new["A_aug2_type"] = trial_hyperparams[2]
        df_new["A_aug2_magnitude"] = trial_hyperparams[3]

        df_new["B_aug1_type"] = trial_hyperparams[4]
        df_new["B_aug1_magnitude"] = trial_hyperparams[5]
        df_new["B_aug2_type"] = trial_hyperparams[6]
        df_new["B_aug2_magnitude"] = trial_hyperparams[7]

        df_new["C_aug1_type"] = trial_hyperparams[8]
        df_new["C_aug1_magnitude"] = trial_hyperparams[9]
        df_new["C_aug2_type"] = trial_hyperparams[10]
        df_new["C_aug2_magnitude"] = trial_hyperparams[11]

        df_new["D_aug1_type"] = trial_hyperparams[12]
        df_new["D_aug1_magnitude"] = trial_hyperparams[13]
        df_new["D_aug2_type"] = trial_hyperparams[14]
        df_new["D_aug2_magnitude"] = trial_hyperparams[15]

        df_new["E_aug1_type"] = trial_hyperparams[16]
        df_new["E_aug1_magnitude"] = trial_hyperparams[17]
        df_new["E_aug2_type"] = trial_hyperparams[18]
        df_new["E_aug2_magnitude"] = trial_hyperparams[19]
        
        df_new["sample_no"] = sample_no
        df_new["mean_late_val_acc"] = reward
        df_new = df_new.round(3)  # round all float values to 3 decimals after point
        df_new["epoch"] = np.arange(1, len(df_new) + 1)
        self.df = pd.concat([self.df, df_new])

    def save(self, trial_no):
        self.df.to_csv(self.args.notebook_path + f"/report_{trial_no}.csv", index=False)

    def add_records_from(self, path_notebook):
        df_notebook = pd.read_csv(path_notebook, comments='#')
        self.df = pd.concat([self.df, df_notebook])

    def get_top_policies(self, k):
        trial_avg_val_acc_df = (
            self.df.drop_duplicates(["trial_no", "sample_no"])
                .groupby("trial_no")
                .mean()["mean_late_val_acc"]
                .reset_index()
        )[["trial_no", "mean_late_val_acc"]]

        x_df = pd.merge(
            self.df.drop(columns=["mean_late_val_acc"]),
            trial_avg_val_acc_df,
            on="trial_no",
            how="left",
        )

        x_df = x_df.sort_values("mean_late_val_acc", ascending=False)

        baseline_val_acc = x_df[x_df["trial_no"] == 0]["mean_late_val_acc"].values[0]

        x_df["expected_accuracy_increase(%)"] = (
                                                        x_df["mean_late_val_acc"] - baseline_val_acc
                                                ) * 100

        self.top_df = x_df.drop_duplicates(["trial_no"]).sort_values(
            "mean_late_val_acc", ascending=False
        )[:k]

        SELECT = [
            "trial_no",
            'A_aug1_type', 'A_aug1_magnitude', 'A_aug2_type', 'A_aug2_magnitude',
            'B_aug1_type', 'B_aug1_magnitude', 'B_aug2_type', 'B_aug2_magnitude',
            'C_aug1_type', 'C_aug1_magnitude', 'C_aug2_type', 'C_aug2_magnitude',
            'D_aug1_type', 'D_aug1_magnitude', 'D_aug2_type', 'D_aug2_magnitude',
            'E_aug1_type', 'E_aug1_magnitude', 'E_aug2_type', 'E_aug2_magnitude',
            "mean_late_val_acc", "expected_accuracy_increase(%)"
        ]
        self.top_df = self.top_df[SELECT]

        print(f"top-{k} policies:", k)
        print(self.top_df)

        return self.top_df

    def output_top_policies(self):
        def get_folder_path(path):
            last = path.split("/")[-1]
            return path.replace(last, "")

        k = len(self.top_df)
        out_path = get_folder_path(self.args.notebook_path) + f"/top{k}_policies.csv"
        self.top_df.to_csv(out_path, index=False)
        print(f"Top policies are saved to {out_path}")


class Objective:
    def __init__(self, args, child_model, notebook):
        self.args = args
        self.child_model = child_model
        self.notebook = notebook
        self.logger = args.logger

    def evaluate(self, trial_no, trial_hyperparams):
        """
        Evaluates objective function
        Trains the child model k times with same augmentation hyperparameters.
        k is determined by the user by `opt_samples` argument.

        Args:
            trial_no (int): no of trial. needed for recording to notebook
            trial_hyperparams (list)
        Returns:
            float: trial-cost = 1 - avg. rewards from samples
        """
        sample_rewards = []
        for sample_no in range(1, self.args.opt_samples + 1):
            self.child_model.trainer.load_checkpoint(0)
            # train
            train_results, valid_results = self.child_model.fit(trial_hyperparams)

            # calculate reward
            reward = self.calculate_reward(train_results, valid_results)
            sample_rewards.append(reward)
            self.notebook.record(
                trial_no, trial_hyperparams, sample_no, reward, (train_results, valid_results)
            )

        trial_cost = 1 - np.mean(sample_rewards)
        self.notebook.save(trial_no)
        print(f"{str(trial_no)}, {str(trial_cost)}, {str(trial_hyperparams)}")
        return trial_cost

    def calculate_reward(self, train_history, valid_history):
        """Calculates reward for the history.

        Reward is mean of largest n validation accuracies which are not overfitting.
        n is determined by the user by `opt_last_n_epochs` argument. A validation
        accuracy is considered as overfitting if the training accuracy in the same
        epoch is larger by 0.05

        Args:
            history (dict): dictionary of loss and accuracy
        Returns:
            float: reward
        """
        df_train_history = pd.DataFrame(train_history)
        df_valid_history = pd.DataFrame(valid_history)

        metric = self.args.reward_metric
        df_valid_history[f"{metric}_overfit"] = df_train_history[f"{metric}"] - df_valid_history[f"{metric}"]
        reward = (
            df_valid_history[df_valid_history[f"{metric}_overfit"] <= 0.10][f"{metric}"]
            .nlargest(self.args.opt_last_n_epochs)
            .mean()
        )
        return reward

