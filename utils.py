import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.figsize'] = (8, 8)
mpl.rcParams['axes.grid'] = False

pretrained_model = tf.keras.applications.MobileNetV2(include_top=True,weights='imagenet')
pretrained_model.trainable = False
decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions
loss_object = tf.keras.losses.CategoricalCrossentropy()

# Helper function to preprocess the image so that it can be inputted in MobileNetV2
def preprocess(image):
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image, (224, 224))
  image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
  image = image[None, ...]
  return image

# Helper function to extract labels from probability vector
def get_imagenet_label(probs):
  return decode_predictions(probs, top=1)[0][0]

def create_adversarial_pattern(input_image, input_label):
  with tf.GradientTape() as tape:
    tape.watch(input_image)
    prediction = pretrained_model(input_image)
    loss = loss_object(input_label, prediction)

  # Get the gradients of the loss w.r.t to the input image.
  gradient = tape.gradient(loss, input_image)
  # Get the sign of the gradients to create the perturbation
  signed_grad = tf.sign(gradient)
  return signed_grad

def createAdversarialExample(image_path):
    #Preprocessing the image
    image_raw = tf.io.read_file(image_path)
    image = tf.image.decode_image(image_raw)
    image = preprocess(image)
    
    #Getting the prediction
    image_probs = pretrained_model.predict(image)
    
    #Getting the index of the class in the imagenet dataset. Will be required later on
    predicted_class_index = tf.argmax(image_probs[0]).numpy()
    
    #Saving the predicted image in the folder image with the name predicted_image
    plt.figure()
    plt.imshow(image[0] * 0.5 + 0.5)  # To change [-1, 1] to [0,1]
    _, image_class, class_confidence = get_imagenet_label(image_probs)
    plt.title('{} : {:.2f}% Confidence'.format(image_class, class_confidence*100))
    filename = 'images/predicted_image.png'
    plt.savefig(filename)

    #Creating the perturbations with respect to the predicted class
    label = tf.one_hot(predicted_class_index, image_probs.shape[-1])
    label = tf.reshape(label, (1, image_probs.shape[-1]))
    perturbations = create_adversarial_pattern(image, label)
    eps = 0.15
    adv_x = image + eps*perturbations
    adv_x = tf.clip_by_value(adv_x, -1, 1)

    #Displaying the perturbed image and also saving it in the folder images with a name adversarial example
    _, label, confidence = get_imagenet_label(pretrained_model.predict(adv_x))
    plt.figure()
    plt.imshow(adv_x[0]*0.5+0.5)
    plt.title('For epsilon {} \n {} : {:.2f}% Confidence'.format(eps,label, confidence*100))
    filename = 'images/adversarial_image.png'
    plt.savefig(filename)

if __name__ == "__main__":
    createAdversarialExample(r'C:\Users\Shubham Srivastava\Desktop\MTech\Thesis\Codes\FGSM\elephant.jpeg')