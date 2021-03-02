import os
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from skimage.transform import resize
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras import backend as K
import numpy as np
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

@app.route("/")
def homepage():
    return render_template('index.html')

@app.route("/about.html")
def aboutus():
    return render_template('about.html')

@app.route("/blog.html")
def blog():
    return render_template('blog.html')

@app.route("/contact.html")
def contact():
    return render_template('contact.html')

@app.route("/reports.html")
def report():
    return render_template('reports.html') 

@app.route("/report1.html")
def report1():
    return render_template('report1.html')

@app.route("/report2.html")
def report2():
    return render_template('report2.html')

@app.route("/report3.html")
def report3():
    return render_template('report3.html')

# Loading the model
print("Loading model")
global sess
sess = tf.compat.v1.Session()
tf.compat.v1.keras.backend.set_session(sess)

global model
#model =load_model('model_test.h5', compile=False)
model = tf.keras.models.load_model('pkmn_all_v9.h5', compile=False)

global graph
graph = tf.compat.v1.get_default_graph()


@app.route("/demos.html", methods=['GET', 'POST'])
def demos_page():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join('static/uploads/', filename))
        return redirect(url_for('prediction', filename=filename))
    return render_template('demos.html')

@app.route('/prediction/<filename>') 
def prediction(filename):    #go to the web page "prediction"
    #Step 1
    img_height = 180
    img_width = 180

    #my_img = plt.imread(os.path.join('uploads', filename))
    #with graph.as_default():
        #tf.compat.v1.keras.backend.set_session(sess)

        #model = tf.keras.models.load_model('model_test.h5', compile=False)

    #Step 2
    #img = resize(my_img, (img_height, img_width, 3))
    picture_path = str('static/uploads/'+filename)

    img = image.load_img(picture_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
        #img_array = img_array/255.0

    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    class_names = ['Abra', 'Aerodactyl', 'Alakazam', 'Arbok', 'Arcanine', 'Articuno', 'Beedrill', 'Bellsprout', 'Blastoise', 'Bulbasaur', 'Butterfree', 'Caterpie', 'Chansey', 'Charizard', 'Charmander', 'Charmeleon', 'Clefable', 'Clefairy', 'Cloyster', 'Cubone', 'Dewgong', 'Diglett', 'Ditto', 'Dodrio', 'Doduo', 'Dragonair', 'Dragonite', 'Dratini', 'Drowzee', 'Dugtrio', 'Eevee', 'Ekans', 'Electabuzz', 'Electrode', 'Exeggcute', 'Exeggutor', 'Farfetchd', 'Fearow', 'Flareon', 'Gastly', 'Gengar', 'Geodude', 'Gloom', 'Golbat', 'Goldeen', 'Golduck', 'Golem', 'Graveler', 'Grimer', 'Growlithe', 'Gyarados', 'Haunter', 'Hitmonchan', 'Hitmonlee', 'Horsea', 'Hypno', 'Ivysaur', 'Jigglypuff', 'Jolteon', 'Jynx', 'Kabuto', 'Kabutops', 'Kadabra', 'Kakuna', 'Kangaskhan', 'Kingler', 'Koffing', 'Krabby', 'Lapras', 'Lickitung', 'Machamp', 'Machoke', 'Machop', 'Magikarp', 'Magmar', 'Magnemite', 'Magneton', 'Mankey', 'Marowak', 'Meowth', 'Metapod', 'Mew', 'Mewtwo', 'Moltres', 'MrMime', 'Muk', 'Nidoking', 'Nidoqueen', 'Nidorina', 'Nidorino', 'Ninetales', 'Oddish', 'Omanyte', 'Omastar', 'Onix', 'Paras', 'Parasect', 'Persian', 'Pidgeot', 'Pidgeotto', 'Pidgey', 'Pikachu', 'Pinsir', 'Poliwag', 'Poliwhirl', 'Poliwrath', 'Ponyta', 'Porygon', 'Primeape', 'Psyduck', 'Raichu', 'Rapidash', 'Raticate', 'Rattata', 'Rhydon', 'Rhyhorn', 'Sandshrew', 'Sandslash', 'Scyther', 'Seadra', 'Seaking', 'Seel', 'Shellder', 'Slowbro', 'Slowpoke', 'Snorlax', 'Spearow', 'Squirtle', 'Starmie', 'Staryu', 'Tangela', 'Tauros', 'Tentacool', 'Tentacruel', 'Vaporeon', 'Venomoth', 'Venonat', 'Venusaur', 'Victreebel', 'Vileplume', 'Voltorb', 'Vulpix', 'Wartortle', 'Weedle', 'Weepinbell', 'Weezing', 'Wigglytuff', 'Zapdos', 'Zubat']
    #predictions = ("This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score)))
        
    print(predictions)

    predictions = {
        "class1":class_names[np.argmax(score)],
        "prob1":"{:.2f}".format(100 * np.max(score)),
        #"picture":picture_path,
      }   
    #Step 5

    url_img_for_html = str("../static/uploads/"+filename)
    print(url_img_for_html)

    path = {
        "url_to_upload":url_img_for_html
    }

    return render_template('predict.html', predictions=predictions, path=path)
    #return render_template('predict.html')
    #return render_template('predict.html')


if __name__ == "__main__":
    app.run()
