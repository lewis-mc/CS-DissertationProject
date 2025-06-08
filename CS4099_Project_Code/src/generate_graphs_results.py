import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import config

def generate_graphs(json_file, json_extra_file, output_dir):

    class_names = ['ABE', 'ART', 'BAS', 'BLA', 'EBO', 'EOS', 'FGC', 'HAC', 'KSC', 'LYI',
                   'LYT', 'MMZ', 'MON', 'MYB', 'NGB', 'NGS', 'NIF', 'OTH', 'PEB', 'PLM', 'PMO']
    # Load the experiment results from the JSON file.
    with open(json_file, 'r') as f:
        experiment_results = json.load(f)

    with open(json_extra_file, 'r') as f:
        experiment_extra_results = json.load(f)

    # Create the output directory if it doesn't exist.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    epochs = []
    train_loss = []
    overall_train_acc =[]
    overall_eval_acc = []


    train_overall_acc = []
    train_accepted_acc = []
    train_overall_bal = []
    train_accepted_bal = []
    eval_overall_acc = []
    eval_accepted_acc = []
    eval_overall_bal = []
    eval_accepted_bal = []
    rejection_rates = []
    num_rejects = []
    train_overall_f1 = []
    train_accepted_f1 = []
    train_overall_recall = []
    train_accepted_recall = []
    train_overall_prec = []
    train_accepted_prec = []
    eval_overall_f1 = []
    eval_accepted_f1 = []
    eval_overall_recall = []
    eval_accepted_recall = []
    eval_overall_prec = []
    eval_accepted_prec = []


    # Helper function (if needed) to average non-zero values.
    def mean_non_zero(values):
        non_zero_values = [v for v in values if v != 0.0]
        return np.mean(non_zero_values) if non_zero_values else 0.0
    
    for epoch_str, results in experiment_results.items():

        if config.hard_negative_mining:

            # For training metrics, assume they are stored under 'train_metrics'
            train_metrics = results.get('train_metrics', {})
            train_overall = train_metrics.get('overall', {})
            train_accepted = train_metrics.get('accepted', {})
            overall_train_acc.append(train_overall.get('accuracy', 0))
            train_accepted_acc.append(train_accepted.get('accuracy', 0))
            train_overall_bal.append(train_overall.get('balanced_accuracy', 0))
            train_accepted_bal.append(train_accepted.get('balanced_accuracy', 0))
            train_overall_f1.append(train_overall.get('f1_score', 0))
            train_accepted_f1.append(train_accepted.get('f1_score', 0))
            train_overall_recall.append(train_overall.get('recall', 0))
            train_accepted_recall.append(train_accepted.get('recall', 0))
            train_overall_prec.append(train_overall.get('precision', 0))
            train_accepted_prec.append(train_accepted.get('precision', 0))
            
            # For evaluation metrics, assume they are stored under 'eval_metrics'
            eval_metrics = results.get('eval_metrics', {})
            eval_overall = eval_metrics.get('overall', {})
            eval_accepted = eval_metrics.get('accepted', {})
            overall_eval_acc.append(eval_overall.get('accuracy', 0))
            eval_accepted_acc.append(eval_accepted.get('accuracy', 0))
            eval_overall_bal.append(eval_overall.get('balanced_accuracy', 0))
            eval_accepted_bal.append(eval_accepted.get('balanced_accuracy', 0))
            eval_overall_f1.append(eval_overall.get('f1_score', 0))
            eval_accepted_f1.append(eval_accepted.get('f1_score', 0))
            eval_overall_recall.append(eval_overall.get('recall', 0))
            eval_accepted_recall.append(eval_accepted.get('recall', 0))
            eval_overall_prec.append(eval_overall.get('precision', 0))
            eval_accepted_prec.append(eval_accepted.get('precision', 0))

        else:

            train_loss.append(results.get('train_loss', 0))
            overall_train_acc.append(results.get('overall_train_accuracy', 0))
            overall_eval_acc.append(results.get('overall_eval_accuracy', 0))

        rejection_rates.append(results.get('rejection_rate', 0))
        num_rejects.append(results.get('num_rejected', 0))

        epoch = int(epoch_str)
        epochs.append(epoch)

        
        
    
    threshold_accuracies = experiment_extra_results.get('threshold_accuracies', {})

    thresholds = list(threshold_accuracies.keys())
    accuracies = [threshold_accuracies[threshold] for threshold in thresholds]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, overall_train_acc, label='Train Overall Accuracy')
    plt.plot(epochs, overall_eval_acc, label='Eval Overall Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Overall Accuracy vs Epochs')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, 'overall_accuracy_vs_epochs.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, accuracies, label='Accuracy')
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.title('Threshold vs Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, 'threshold_vs_accuracy.png'))
    plt.close()

    if config.run_mode == 'test':

        final_conf_matrix = np.array(experiment_extra_results['final_test_metrics']['overall']['confusion_matrix'])
        plt.figure(figsize=(12, 10))
        sns.heatmap(final_conf_matrix, annot=True, cmap='Blues', fmt='.2f',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix')
        plt.savefig(os.path.join(output_dir, f'overall_test_confusion_matrix_epoch.png'))
        plt.close()
        
        final_conf_matrix = np.array(experiment_extra_results['final_test_metrics']['accepted']['confusion_matrix'])
        plt.figure(figsize=(12, 10))
        sns.heatmap(final_conf_matrix, annot=True, cmap='Blues', fmt='.2f',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix')
        plt.savefig(os.path.join(output_dir, f'accepted_test_confusion_matrix_epoch.png'))
        plt.close()

    final_conf_matrix = np.array(experiment_extra_results['final_validation_metrics']['overall']['confusion_matrix'])
    plt.figure(figsize=(12, 10))
    sns.heatmap(final_conf_matrix, annot=True, cmap='Blues', fmt='.2f',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix')
    plt.savefig(os.path.join(output_dir, f'overall_eval_confusion_matrix_epoch.png'))
    plt.close()

    final_conf_matrix = np.array(experiment_extra_results['final_validation_metrics']['accepted']['confusion_matrix'])
    plt.figure(figsize=(12, 10))
    sns.heatmap(final_conf_matrix, annot=True, cmap='Blues', fmt='.2f',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix')
    plt.savefig(os.path.join(output_dir, f'accepted_eval_confusion_matrix_epoch.png'))
    plt.close()