from numpy import np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
import importlib, cv2, os, random, psutil, time,  getpass, platform, csv

class SuperviseTrainer:

    def __init__(self, export_directory, model_file, utils_file, data_directory=""):
        self.export_directory = export_directory
        self.data_directory = data_directory
        self.datasets = None # provides ability to combine datasets
        self.model_file = importlib.import_module(model_file)
        self.utils = importlib.import_module(utils_file)

        self.tensorboard = None

        self.X_training = None
        self.Y_training = None
        self.X_test = None
        self.Y_test = None

    def build_model(self):

        # Setup tensorboard for viewing model development
        time_stamp = time.time()
        path = os.path.dirname(os.path.realpath(__file__))  # get python file path
        self.tensorboard = TensorBoard(log_dir="{}/logs/{}".format(path, time_stamp))
        print("Run `tensorboard --logdir=\"{}/logs/{}\"` and see `http://localhost:6006` to see training status and graph".format(path,time_stamp) + "\n\n")

        self.model_file.model.summary()  # print a summary representation of model

        self.model_file.compile_model()

        print("Model compiled!")

    def set_dataset(self):
        data_dir = input(
            "Enter root dataset directory (i.e. ~/racecarDatasets). If left blank the USB root dataset directory will be used.")
        if data_dir.strip() == "":
            if platform.system() == 'Windows':
                data_dir = "E:\\"  # directory of expected USB flash drive on Windows
                print("Windows detected! Searching " + data_dir + " for dataset")
            else:
                data_dir = "/media/" + getpass.getuser() + "/racecarDataset"  # directory of expected USB flash drive on Linux
                print("Linux detected! Searching " + data_dir + " for dataset")

        print("Dataset Location: " + data_dir + "\n")

        # This function lets you combine ex. dataset with dataset1 and dataset2 for a single training session. This makes adding in more data easy and separates data collection periods
        special_dataset_read = input(
            "Would you like to combine multiple datasets in the root dataset directory for training? (y/n)")
        if special_dataset_read.lower().strip() == 'y':
            print("The dataset root should be configured to cotnain files in the format " + os.path.join(
                data_dir + "dataset") + ",1,2,3,4,...")
            datasets = input(
                "Enter in the sufixes for the datasets to read from: (ex.  `,3,5`  will read from " + os.path.join(
                    data_dir + "dataset") + ", " + os.path.join(data_dir + "dataset") + "3, and " + os.path.join(
                    data_dir + "dataset") + "5)").split(',')
            for i, suffix in enumerate(datasets):
                datasets[i] = "dataset" + suffix
        else:
            datasets = ["dataset"]
            print("Using default " + os.path.join(data_dir + "dataset") + " for data")

        print("Datasets to read from: " + str(datasets) + "\n\n")

        self.datasets = datasets

        print("Reading sample image")

        dataset_index = random.randint(len(self.datasets), size=1)
        color_img = mpimg.imread(random.choice(os.listdir(os.path.join(self.datasets[dataset_index], "color_images/"))))
        depth_img = mpimg.imread(random.choice(os.listdir(os.path.join(self.datasets[dataset_index], "depth_images/"))))
        plt.imshow(color_img)
        plt.imshow(depth_img)
        plt.show()

    def load_dataset(self, dynamic=True, percent_training=80, existing_npy=None, npy_save_directory=None):
        # load training set
        print("Loading training set...")

        if dynamic:
            print("Loading data file paths for dynamic loading during training/evaluation")
            # these will be 2D arrays where each row represents a dataset
            x = []
            y = []
            for dataset in self.datasets:
                with open(os.path.join(os.path.join(self.data_directory, dataset), "tags.csv")) as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        # print(row['Time_stamp'] + ".jpg", row['Steering_angle'])
                        x.append(row['time_stamp'] + ".jpg",)  # get image path
                        y.append(float(row['raw_steering']),) # get steering value

                print("Number of data samples is " + len(y))

            data = list(zip(x,y))
            random.shuffle(data)
            x,y = zip(*data)