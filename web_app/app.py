import os
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from transformers import AutoImageProcessor, Dinov2ForImageClassification, Dinov2Config
import cv2

# Initialize the Flask application
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize ResNet model
resnet_model = models.resnet101()
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_ftrs, 3)
device = torch.device("cpu")
resnet_model.load_state_dict(torch.load('resnet.pth', map_location=torch.device('cpu')),)
resnet_model = resnet_model.to(device)
resnet_model.eval()

# Initialize DinoV2 model
dinov2_model = Dinov2ForImageClassification.from_pretrained('dinov2-large', num_labels=3)
dinov2_model.load_state_dict(torch.load('dinov2.pth', map_location=torch.device('cpu')))
dinov2_model.eval()

# Image preprocessing for ResNet
preprocess_res = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Image preprocessing for DinoV2
preprocess_dino = transforms.Compose([
    transforms.Resize(518),
    transforms.CenterCrop(518),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Classify image and return the class_id and output probabilities
def classify_image(image_path, model_name):
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img_t = None    
    if model_name == 'ResNet':    
        img_t = preprocess_res(img)
    elif model_name == 'DinoV2':
        img_t = preprocess_dino(img) 
    else:
        raise ValueError(f"Unknown model name: {model_name}")       
    batch_t = torch.unsqueeze(img_t, 0)
    with torch.no_grad():
        out = None
        if model_name == 'ResNet':
            out = resnet_model(batch_t)
        elif model_name == 'DinoV2':
            out = dinov2_model(batch_t).logits
        else:
            raise ValueError(f"Unknown model name: {model_name}")      
    probabilities = torch.nn.functional.softmax(out[0], dim=0)
    top_prob, top_catid = torch.topk(probabilities, 1)
    print(probabilities)
    tensor_path = os.path.join(app.config['UPLOAD_FOLDER'], 'preprocessed_' + os.path.basename(image_path) + '.pt')
    torch.save(img_t, tensor_path)
    return top_catid.item(), probabilities.tolist()

@app.route('/', methods=['GET', 'POST'])
#@auth.login_required
def upload_file():
    if request.method == 'POST':
        model_name = request.form.get('model')

        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Save the uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Classify the image
            class_id, probabilities = classify_image(file_path, model_name)  # Pass model_name to classify_image
            class_names = ['group_1', 'group_2', 'group_3']
            class_name = class_names[class_id]
            
            return render_template('result.html', class_id=class_id, class_name=class_name, probabilities=probabilities, filename=filename, model_name=model_name)
    
    return render_template('upload.html')


@app.route('/generate_gradcam/<filename>', methods=['GET'])
def generate_gradcam(filename):
    try:
        model_name = request.args.get('model')
        class_id = request.args.get('class_id')
        
        # Load the preprocessed image tensor
        tensor_path = os.path.join(app.config['UPLOAD_FOLDER'], 'preprocessed_' + filename + '.pt')
        img_t = torch.load(tensor_path).unsqueeze(0)
       
        # Convert the tensor back to the original image format for visualization
        img = img_t.squeeze().permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)

        if model_name == 'ResNet':
            target_layers = [resnet_model.layer4[-1]]
            cam = GradCAM(model=resnet_model, target_layers=target_layers)
        elif model_name == 'DinoV2':
            target_layers = [dinov2_model.classifier]
            cam = GradCAM(model=dinov2_model, target_layers=target_layers)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        # Define the target class (Change the class index as needed for your specific task)
        targets = [ClassifierOutputTarget(int(class_id))]  # Adjust target class index for your resnet_model's specific class

        # Generate the Grad-CAM heatmap
        grayscale_cam = cam(input_tensor=img_t, targets=targets, aug_smooth=True)

        # In this example grayscale_cam has only one image in the batch
        grayscale_cam = grayscale_cam[0, :]

        # Visualize the heatmap
        visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)

        # Save the Grad-CAM image
        gradcam_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'gradcam_' + filename)
        cv2.imwrite(gradcam_image_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

        return 'gradcam_' + filename
    except Exception as e:
        print(f"Error generating Grad-CAM image: {e}")
        return '', 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', port=port)  # Ensure we are running on the correct port