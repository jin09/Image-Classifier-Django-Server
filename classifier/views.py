from django.http import HttpResponse
import os, sys
import tensorflow as tf
import base64
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def index(request):
    """
        if request.method == "POST":
            imagebase64 = request.POST.get("encoded_string")
            imagebase64 = base64.b64encode(imagebase64.encode())
            imagename = request.POST.get("image_name")
            pwd = os.getcwd()
            pwd += "/test.jpg"
            print (pwd)
            try:
                os.remove(pwd)
            except:
                print("no image with name test found to be deleted !!")
            with open(pwd, "wb") as fh:
                fh.write(base64.decodebytes(imagebase64))
            return HttpResponse(classify(pwd))
    """
    if request.method == "GET":
        return HttpResponse(classify('/home/gautam/Desktop/server/classifier/try.jpg'))
        # return HttpResponse("Hello, world. You're at the polls index.")
    if request.method == 'POST':
        try:
            uploaded_file = request.FILES['file']
            file_name = "/testing.jpg"
            # Write content of the file chunk by chunk in a local file (destination)
            pwd = os.getcwd()
            pwd += file_name
            try:
                os.remove(pwd)
            except:
                print("no image with name test found to be deleted !!")
            with open(pwd, 'wb+') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)
            return HttpResponse(classify(pwd))
        except:
            return HttpResponse('ERROR')


def classify(path):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    label = ""
    scor = ""
    # change this as you see fit
    image_path = path

    # Read in the image_data
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("/home/gautam/Desktop/server/classifier/retrained_labels.txt")]

    # Unpersists graph from file
    with tf.gfile.FastGFile("/home/gautam/Desktop/server/classifier/retrained_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        predictions = sess.run(softmax_tensor, \
                               {'DecodeJpeg/contents:0': image_data})

        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        for node_id in top_k:
            human_string = label_lines[node_id]
            label = human_string
            score = predictions[0][node_id]
            scor = score
            break
    if scor > 0.9:
        return label
    else:
        return "Fruit not found"
