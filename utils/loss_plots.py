import csv
import matplotlib.pyplot as plt

def load_avg_loss_per_epoch(csv_path):
    """
    Loads the CSV file at csv_path and returns a list of average losses
    per epoch in ascending order of epoch.
    """
    losses_by_epoch = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip the header
        
        for row in reader:
            # row format:
            # epoch, itr, loss, loss-jepa, reg-loss, enc-grad-norm, pred-grad-norm, gpu-time(ms), wall-time(ms)
            epoch = int(row[0])
            loss = float(row[2])
            
            if epoch not in losses_by_epoch:
                losses_by_epoch[epoch] = []
            losses_by_epoch[epoch].append(loss)

    # Compute the average loss for each epoch
    avg_loss_per_epoch = []
    for epoch in sorted(losses_by_epoch.keys()):
        epoch_losses = losses_by_epoch[epoch]
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_loss_per_epoch.append(avg_loss)

    return avg_loss_per_epoch

def main():
    # Update these paths as needed
    train_filename = "/home/madhavan/jepa/logging/full_run_subset_full_r0.csv"
    # eval_filename = "/home/madhavan/jepa/logging/full_run_subset_full_eval_r0.csv"

    # Load the average losses
    avg_loss_per_epoch_train = load_avg_loss_per_epoch(train_filename)
    # avg_loss_per_epoch_eval = load_avg_loss_per_epoch(eval_filename)

    # Set up epochs for plotting
    # Note: If you have different epoch counts in train vs eval, 
    # you can handle that logic however you wish. 
    train_epochs = range(1, len(avg_loss_per_epoch_train) + 1)
    # eval_epochs = range(1, len(avg_loss_per_epoch_eval) + 1)

    # Plot both curves on the same figure
    plt.figure(figsize=(8, 5))
    plt.plot(train_epochs, avg_loss_per_epoch_train, label="Train")
    # plt.plot(eval_epochs, avg_loss_per_epoch_eval, label="Eval")
    # plt.title("Average Loss Per Epoch: Train vs Eval")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.legend()  # show legend to distinguish the two lines

    # Save the figure (you could also call plt.show() if you want to display)
    plt.savefig(f"/home/madhavan/jepa/logging/small_data.png")
    # Uncomment the next line if you want to display the plot interactively:
    # plt.show()

if __name__ == "__main__":
    main()
