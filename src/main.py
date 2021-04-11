# coding=utf-8
import base64
import io

import numpy
from PIL import Image
from flask import Flask, send_file, request, json, jsonify

from src.utils import *

application = Flask(__name__)
latent_space_size = 64
image_height = 256
image_width = 768


@application.route('/get')
def get_output():
    argument = request.args.get('values', None, str)
    if argument is None:
        input_values = create_noise(1, latent_space_size).to(device)
    else:
        data = numpy.array(json.loads(argument)).astype(numpy.float32)
        input_values = torch.tensor(numpy.reshape(data, (1, latent_space_size, 1, 1))).to(device)
    file_object = image_from_values(input_values)
    response = send_file(file_object, 'image/png')
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

@application.route('/getMultiple')
def get_outputs():
    argument = request.args.get('values', None, str)
    values = numpy.array(json.loads(argument), numpy.float32)
    images = []
    input_values = torch.tensor(numpy.reshape(values, (len(values), latent_space_size, 1, 1))).to(device)
    imagesData = image_from_values(input_values)
    for data in imagesData:
        bytes = base64.b64encode(data.getvalue())
        base64_string = bytes.decode('ascii')
        images.append(base64_string)
    response = jsonify(images)
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

def image_from_values(input_values):
    tensor = generator(input_values).cpu().detach()
    ndarr = tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to(device, torch.uint8).cpu().numpy()
    ndarr = ndarr[:,:,:,[2,1,0]]
    files = []
    for data in ndarr:
        img = Image.fromarray(data)
        file_object = io.BytesIO()
        img.save(file_object, 'PNG')
        file_object.seek(0)
        files.append(file_object)
    if len(files) > 1:
        return files
    else:
        return files[0]

@application.route('/canvas', methods=['POST'])
def get_canvas():
    file_image = request.files.get('image')
    if file_image is None:
        return
    else:
        pil_image = Image.open(file_image).convert('RGB')
        pil_image = pil_image.resize((image_width, image_height))
        data = numpy.array(pil_image).astype(numpy.float32)
        values = numpy.reshape(data, (1, image_width, image_height, 3))
        values = values[:,:,:,[2,1,0]]
        values = numpy.transpose(values, (0, 3, 1, 2))
        values = values/255.0
        input_values = torch.tensor(numpy.reshape(values, (1, 3, image_height, image_width))).to(device)
    image = encoder(input_values).view(latent_space_size).cpu().detach().numpy().tolist()
    response = jsonify(image)
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print('You are using your GPU')
    else:
        device = torch.device("cpu")
    loadedGenerator = torch.load("generator.pth", map_location=device)
    if isinstance(loadedGenerator, dict):
        generator = loadedGenerator["model"]
    else:
        generator = loadedGenerator
    generator.eval()
    print("generator:" + generator)

    loadedEncoder = torch.load("encoder.pth", map_location=device)
    if isinstance(loadedEncoder, dict):
        encoder = loadedEncoder["model"]
    else:
        encoder = loadedEncoder
    encoder.eval()
    application.run()
