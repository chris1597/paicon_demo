# PAICON Demo

This repository contains the training/validation scripts and a web app for a trained ResNet and DINOv2 model.

## Training

Training was performed on Google Colab, but it can be adapted to run on your local machine by setting the data path to the root path of the `group_1`, `group_2`, and `group_3` folders.

## Web App Deployment

You can deploy the web app on your local machine.

1. **Download DINOv2 Model**
   - [DINOv2 Model](https://drive.google.com/file/d/17yb6Drdt_RKQgmTrA9aAf3EQox2lpiqS/view?usp=drive_link)
   - Place the DINOv2 model in the root folder.

2. **Download ResNet Model**
   - [ResNet Model](https://drive.google.com/file/d/1-0l1mjj3UGm1FGW0hcE7ibcfES87aXYk/view?usp=drive_link)
   - Place the ResNet model in the root folder.

3. **Download SafeTensor File**
   - [SafeTensor File](https://drive.google.com/file/d/1HRTnHojH_prAdfWv6XdjuMhiF-0IpbRk/view?usp=drive_link)
   - Place the SafeTensor file inside the `dinov2-large` folder.

### Web App

The inference web app is live and can be accessed [here](https://l84csss.94.130.73.96.sslip.io/).

### Requirements

The `requirements.txt` file contains the necessary dependencies to run the web app and the training script. Additional dependencies for the validation script were installed on Colab.

## Results

### Resnet Training Results

![Resnet Training Results](assets/resnet_res.png)

### DinoV2 Training Results

![Dino Training Results](assets/dinov2_res.png)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/paicon_demo.git
    cd paicon_demo
    ```

2. Install the dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Run the web app:
    ```sh
    cd web_app
    python app.py
    ```

3. Run the training:
    ```sh
    cd classifier
    python train.py
    ```
