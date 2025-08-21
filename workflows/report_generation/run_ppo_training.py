from RewardingVisualDoubt import training

from radialog_report_generation_ppo_training import train

CONFIG_FILE = "/home/guests/deniz_gueler/repos/RewardingVisualDoubt/workflows/report_generation/configs/quadratic_reward_config.yaml"

if __name__ == "__main__":
    metaparamaters, hyperparameters = training.load_default_configs(CONFIG_FILE)
    train(metaparamaters, hyperparameters)
