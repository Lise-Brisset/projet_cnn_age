# librairies 
import logging

# Configuration du logging
logging.basicConfig(
    filename='/home/lise/Documents/TAL_M2S3/CNN/log_vectorisation.log', # Essayez d'utiliser un chemin absolu ici
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Début du script de vectorisation")


import glob 
import speechbrain as sb
import pandas as pd
import torch
import os
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
import time



# Démarrer le timer
start_time = time.time()



# Ouverture fichiers

chemin_base = "/media/lise/SMARTDISK/DufourOrane_Memoire-M2TAL/Corpus4Sec/ESLO_4Sec" 

#chemin_base = "./projet_cnn_age/data" 

list_fichier = glob.glob(f"{chemin_base}/ESLO2_*/*_16000Hz.wav")
print(list_fichier[:10])

# On récupère le csv contenant les infos sur les locuteurs : 
metadata_fichier = pd.read_csv("/home/lise/Documents/TAL_M2S3/CNN/projet_cnn_age/data/metadonnees_ESLO2_ENT_ENTJEUN.csv")


# rajout classses d'age
def class_age(age):
	if age == 'vieux' or int(age) > 60: 
		return "vieux"
	elif int(age) < 30:
		return "jeune"
	else:
		return "mid"
	
metadata_fichier['classe_age'] = metadata_fichier['age'].apply(class_age)



# vectorisation 
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb")

def vectorisation(fichier):
	signal, fs = torchaudio.load(fichier)
	embeddings = classifier.encode_batch(signal)
	return embeddings


logging.info("Début de la vectorisation")
logging.info(f"Nombre de fichiers à traiter: {len(list_fichier)}")


# enregistrement des vecteurs dans leur bon directory (selon classe d'age et sexe) 
for i, fichier in enumerate(list_fichier): 
    try:
        embeddings = vectorisation(fichier)
        directory_name = fichier.split("/")[-2]
        reference_name = fichier.split('/')[-1].replace('.wav', '')

        # Vérification de l'existence du répertoire dans les métadonnées
        if directory_name in metadata_fichier['directory'].values:
            age = metadata_fichier[metadata_fichier['directory'] == directory_name]['classe_age'].values[0]
            sexe = metadata_fichier[metadata_fichier['directory'] == directory_name]['sexe'].values[0]
            logging.info(f"Traitement du fichier n.{i}: {fichier}, Age: {age}, Sexe: {sexe}, Référence: {reference_name}")

            # Création du répertoire si nécessaire
            target_directory = f"/media/lise/SMARTDISK/BrissetLise_M2TAL_age/xvectors_4sec/{sexe}/{age}/{directory_name}"
            if not os.path.exists(target_directory):
                os.makedirs(target_directory)
                logging.info(f"Création du répertoire: {target_directory}")

            chemin = f"{target_directory}/{reference_name}.pt"
            
            # Sauvegarde des embeddings
            torch.save(embeddings, chemin)
            logging.info(f"Embeddings sauvegardés dans: {chemin}")
        else:
            logging.warning(f"Répertoire non trouvé dans les métadonnées: {directory_name}")

    except Exception as e:
        logging.error(f"Erreur lors du traitement du fichier {fichier}: {e}")



# Fin du script
logging.info(f"Fin du script, durée d'exécution: {time.time() - start_time} secondes")

