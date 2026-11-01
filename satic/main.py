from falsk import Flask
import random2
app = Flask(__name__)
@app.route('/')
def code_genration():
    random_final=""
    for i in range(0,5):
        random_number = random2.randint(0,9)
        random_final=str(random_number)+random_final
    print("Random number generated is:",random_final)

