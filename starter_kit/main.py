from typing import Union, Dict
import argparse
from src.trainer.trainer import Trainer
import os
import torch
import optuna
import gc
from optuna.trial import *

def objective(trial: optuna.trial.Trial, params: Dict[str, Union[str, int, float, bool, None]]):
    params["batch_size"] = trial.suggest_int(name="batch_size", low=8, high=64, step=8)
    params["num_epochs"] = trial.suggest_int(name="num_epochs", low=1, high=10, step=1)
    params["lr"] = trial.suggest_categorical(name="lr", choices=[1e-6, 5e-6, 8e-6, 9e-6, 1e-5, 1.5e-5, 2e-5, 2.5e-5, 3e-5, 4e-5, 5e-5, 8e-5, 1e-4])
    params["wd"] = trial.suggest_categorical(name="wd", choices=[1e-6, 5e-6, 8e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1])
    params["dropout"] = trial.suggest_float(name="dropout", low=0.0, high=0.5, step=0.1)
    params["warmup_pcrt"] = trial.suggest_float(name="warmup_pcrt", low=0.01, high=0.1, step=0.01)
    params["optuna"] = trial
    params: Dict[str, Union[str, int, float, bool, optuna.trial.Trial, None]]
    
    trainer = Trainer(params=params)
    score = trainer.evaluation_loop(partition="dev", print_results=False)["f1_weighted"]
    gc.collect()
    torch.cuda.empty_cache()
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="main", description="Assessing Logical Coherence of LLMs via Fine-Grained NLI")
    
    parser.add_argument("--model_path", type=str, help="A path to a directory containing model weights saved using save_pretrained() method or a model id of a pretrained model hosted inside a model repo on huggingface.co.", required=True)
    parser.add_argument("--tokenizer_path", type=str, help="A path to a directory containing vocabulary files required by the tokenizer saved using the save_pretrained() method or a the model id of a predefined tokenizer hosted inside a model repo on huggingface.co.", required=True)
    
    parser.add_argument("--do_train", action=argparse.BooleanOptionalAction, help="Whether to perform training.", required=True)
    parser.add_argument("--train_path", type=str, help="Path to the training dataset.")
    parser.add_argument("--do_dev", action=argparse.BooleanOptionalAction, help="Whether to evaluate on a development dataset.", required=True)
    parser.add_argument("--dev_path", type=str, help="Path to the development dataset.")
    parser.add_argument("--do_test", action=argparse.BooleanOptionalAction, help="Whether to perform testing.", required=True)
    parser.add_argument("--test_path", type=str, help="Path to the testing dataset.")

    parser.add_argument("--seed", type=int, help="Random seed for reproducibility.", required=True)
    parser.add_argument("--max_length", type=int, help="Maximum length to use by one of the truncation/padding parameters.", required=True)
    parser.add_argument("--dropout", type=float, help="Dropout rate (between 0 and 0.5).")
    parser.add_argument("--warmup_pcrt", type=float, help="Percentage of training steps to perform linear learning rate warmup for (between 0 and 0.1 and a step of 0.01).")
    parser.add_argument("--lr", type=float, help="Learning rate for the optimizer ([1e-6, 5e-6, 8e-6, 9e-6, 1e-5, 1.5e-5, 2e-5, 2.5e-5, 3e-5, 4e-5, 5e-5, 8e-5, 1e-4]).")
    parser.add_argument("--batch_size", type=int, help="Batch size for training and evaluation (multiple of 8 and less than 64).")
    parser.add_argument("--wd", type=float, help="Weight decay for the optimizer. ([1e-6, 5e-6, 8e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1])")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs (between 1 and 10).")

    parser.add_argument("--is_optuna_trial", action=argparse.BooleanOptionalAction, help="Whether this run is an Optuna hyperparameter optimization trial.", required=True)
    parser.add_argument("--study_name", type=str, help="Optuna study name")
    parser.add_argument("--storage_name", type=str, help="Optuna storage name")
    parser.add_argument("--n_trials", type=int, help="Optuna number of trials")
    
    parser.add_argument("--save_path", type=str, help="Path to save the trained model.")
    parser.add_argument("--epoch_to_stop", type=int, help="Epoch number to stop training in case intermediate saving is desired.")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory where timestamped prediction CSV files will be written.")
    parser.add_argument("--reference_path", type=str, help="Optional hidden reference CSV for official DiCo-NLI scoring.")

    args = parser.parse_args()

    if not args.do_train and not args.do_test:
        raise Exception("At least one of --do_train or --do_test must be True.")
    
    if args.do_train:
        assert args.train_path is not None and os.path.isfile(path=args.train_path), "A valid --train_path must be specified when --do_train is True."
        if not args.is_optuna_trial:
            assert all(arg is not None for arg in [args.dropout, args.warmup_pcrt, args.lr, args.batch_size, args.wd, args.num_epochs]), "When --do_train is True, --dropout, --warmup_pcrt, --lr, --batch_size, --wd and --num_epochs must be specified."
            assert 0 <= args.dropout <= 0.5, "--dropout must be between 0 and 0.5."
            assert 0 <= args.warmup_pcrt <= 0.1, "--warmup_pcrt must be between 0 and 0.1."
            assert args.lr in [1e-6, 5e-6, 8e-6, 9e-6, 1e-5, 1.5e-5, 2e-5, 2.5e-5, 3e-5, 4e-5, 5e-5, 8e-5, 1e-4], "Learning rate must be one of the following: [1e-6, 5e-6, 8e-6, 9e-6, 1e-5, 1.5e-5, 2e-5, 2.5e-5, 3e-5, 4e-5, 5e-5, 8e-5, 1e-4]"
            assert args.wd in [1e-6, 5e-6, 8e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1], "Weight decay must be one of the following: [1e-6, 5e-6, 8e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]"
            assert 1 <= args.num_epochs <= 10, "--num_epochs must be between 1 and 10."
            assert 8 <= args.batch_size <= 64 and args.batch_size % 8 == 0, "--batch_size must be a multiple of 8 and less than or equal to 64."
        if args.do_dev:
            assert args.dev_path is not None and os.path.isfile(path=args.dev_path), "A valid --dev_path must be specified when --do_dev is True."

    if args.do_dev and not args.do_train:
        raise Exception("--do_dev can only be True if --do_train is also True.")
    
    if args.do_test:
        assert args.test_path is not None and os.path.isfile(path=args.test_path), "A valid --test_path must be specified when --do_test is True."
        assert 8 <= args.batch_size <= 64 and args.batch_size % 8 == 0, "--batch_size must be a multiple of 8 and less than or equal to 64."
    
    if any(arg is not None for arg in [args.num_epochs, args.lr, args.wd, args.warmup_pcrt]):
        assert args.do_train, "Some hyperparameters were set but training is disabled"  

    if args.is_optuna_trial:
        assert args.do_train and args.do_dev, "When --is_optuna_trial is True, both --do_train and --do_dev must be True."
        assert all(arg is None for arg in [args.dropout, args.warmup_pcrt, args.lr, args.batch_size, args.wd, args.num_epochs]), "When --is_optuna_trial is True, --dropout, --warmup_pcrt, --lr, --batch_size, --wd and --num_epochs must not be specified."
        assert args.save_path is None, "--save_path must not be specified when --is_optuna_trial is True."
        assert args.epoch_to_stop is None, "--epoch_to_stop must not be specified when --is_optuna_trial is True."
        assert not args.do_test, "--do_test must not be specified when --is_optuna_trial is True."
        assert args.test_path is None, "--test_path must not be specified when --is_optuna_trial is True."
        assert args.study_name is not None, "--study_name must be specified when --is_optuna_trial is True."
        assert args.storage_name is not None, "--storage_name must be specified when --is_optuna_trial is True."
        assert args.n_trials is not None and args.n_trials > 0, "--n_trials must be a positive integer when --is_optuna_trial is True."
    
    if args.study_name is not None or args.storage_name is not None or args.n_trials is not None:
        assert args.is_optuna_trial, "--study_name, --storage_name and --n_trials can only be set if --is_optuna_trial is True."
    if args.save_path is not None:
        assert os.path.isdir(s=os.path.dirname(p=args.save_path)), "--save_path must be a valid path to a directory."
        assert not args.is_optuna_trial, "--save_path must not be set when --is_optuna_trial is True."
        assert args.do_train, "--save_path can only be set if --do_train is True."  
    if args.epoch_to_stop is not None:
        assert 1 <= args.epoch_to_stop <= args.num_epochs, "--epoch_to_stop must be between 1 and num_epochs."
        assert not args.is_optuna_trial, "--epoch_to_stop not be set when --is_optuna_trial is True."
        assert args.do_train, "--epoch_to_stop can only be set if --do_train is True."  

    params: Dict[str, Union[str, int, float, bool, None]] = {
        "model_path": args.model_path,
        "tokenizer_path": args.tokenizer_path,
        "do_train": args.do_train,
        "train_path": args.train_path,
        "do_dev": args.do_dev,
        "dev_path": args.dev_path,
        "do_test": args.do_test,
        "test_path": args.test_path,
        "seed": args.seed,
        "dropout": args.dropout,
        "warmup_pcrt": args.warmup_pcrt,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "wd": args.wd,
        "num_epochs": args.num_epochs,
        "save_path": args.save_path,
        "epoch_to_stop": args.epoch_to_stop,
        "results_dir": args.results_dir,
        "reference_path": args.reference_path,
        "max_length": args.max_length,
    }
    
    if args.is_optuna_trial:
        func = lambda trial: objective(trial=trial, params=params)
        study = optuna.create_study(study_name=args.study_name, storage=args.storage_name, direction="maximize", load_if_exists=False)
        # study = optuna.load_study(study_name=args.study_name, storage=args.storage_name)
        study.optimize(func=func, n_trials=args.n_trials, gc_after_trial=True)

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("Number of finished trials: ", len(study.trials))
        print("Number of pruned trials: ", len(pruned_trials))
        print("Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("Value: ", trial.value)

        print("Params:")
        for key, value in trial.params.items():
            print("{}: {}".format(key, value))
    else:
        params["optuna"] = None
        Trainer(params=params)
