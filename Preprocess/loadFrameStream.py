import os 
import random
# Dataset Path 



def sendVideoTrain(ap,rp):

    PATH_Attack = ap
    PATH_Real = rp



    # Load Videos 
    attack = os.listdir(PATH_Attack)
    real = os.listdir(PATH_Real)

    attack_video = []
    real_video = []

    for file in attack:
        f = os.path.join(attack,file)
        attack.append(f)

    for file in real:
        r = os.path.join(real,file)
        real_video.append(r)

    random.shuffle(attack_video)
    random.shuffle(real_video)
    r_int = random.random()
    if r_int > 0.5:
        r1 = random.choice(real_video)

        return  real,True
    else: 
        r2 = random.choice(attack_video)
        return r2,False
    


