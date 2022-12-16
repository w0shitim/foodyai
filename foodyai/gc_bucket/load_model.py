#due to its size we decided to save the model and its config file
#on pur google cloud storage bucket

from google.cloud import storage

from colorama import Fore, Style

def get_model(download_to_disk = True,
              destination_file_name = './raw_data/model_final.pth'):

    '''
    import model from gc strorage bucket
    either download it to package or not

    /!\ the raw_data is git ignored. Feel free to modify the path or git ignore file
    '''

    #print(Fore.BLUE + "Getting the model from the google storage bucket")

    BUCKET_NAME = "foodygs"
    blob_name = "foodyai_data/model_final/model_final.pth"

    storage_client = storage.Client()

    bucket = storage_client.bucket(BUCKET_NAME)

    blob = bucket.blob(blob_name)

    if download_to_disk == True:

        blob.download_to_filename(destination_file_name)
        print(Fore.BLUE +
            "Downloaded storage object {} from bucket {} to local file {}.".format(
            blob_name, BUCKET_NAME, destination_file_name)
        )

    model = ''

    if download_to_disk == False:
        print(Fore.BLUE + f"retrieving {blob_name} from gcloud")
        model = blob.download_as_string()
        print(Fore.GREEN + f"received... test")

    #print(Fore.GREEN + "Model retrieve, good to go")

    return model

def get_config(download_to_disk = True,
              destination_file_name = './raw_data/config.yml'):

    '''
    import config from gc strorage bucket
    either download it to package or not

    /!\ the raw_data is git ignored. Feel free to modify the path or git ignore file
    '''

    #print(Fore.BLUE + "Getting the model from the google storage bucket")

    BUCKET_NAME = "foodygs"
    blob_name = "foodyai_data/model_final/config.yml"

    storage_client = storage.Client()

    bucket = storage_client.bucket(BUCKET_NAME)

    blob = bucket.blob(blob_name)

    if download_to_disk == True:

        blob.download_to_filename(destination_file_name)
        print(Fore.GREEN +
            "Downloaded storage object {} from bucket {} to local file {}.".format(
            blob_name, BUCKET_NAME, destination_file_name
        )
    )
    config = ''

    if download_to_disk == False:
        print(Fore.BLUE +f"retrieving {blob_name} from gcloud")
        config = blob.download_as_string()
        print(Fore.GREEN +f"received... test")

    #print(Fore.GREEN + "Config retrieve, good to go")

    return config
