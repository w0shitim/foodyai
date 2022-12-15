#get class_to_category.json
#necessary file for prediction

from google.cloud import storage

from colorama import Fore, Style

def get_class(download_to_disk = True,
              destination_file_name = './raw_data/class_to_category.json'):

    '''
    import class_to_category.json from gc strorage bucket
    either download it to package or not

    /!\ the raw_data is git ignored. Feel free to modify the path or git ignore file
    '''

    #print(Fore.BLUE + "Getting the model from the google storage bucket")

    BUCKET_NAME = "foodygs"
    blob_name = "foodyai_data/class_to_category.json"

    storage_client = storage.Client()

    bucket = storage_client.bucket(BUCKET_NAME)

    blob = bucket.blob(blob_name)

    if download_to_disk == True:

        blob.download_to_filename(destination_file_name)
        print(Fore.BLUE +
            "Downloaded storage object {} from bucket {} to local file {}.".format(
            blob_name, BUCKET_NAME, destination_file_name)
        )

    class_to_cat = ''

    if download_to_disk == False:
        print(Fore.BLUE + f"retrieving {blob_name} from gcloud")
        class_to_cat = blob.download_as_string()
        print(Fore.GREEN + f"received... test")

    #print(Fore.GREEN + "Model retrieve, good to go")

    return class_to_cat

def get_annotations(download_to_disk = True,
              destination_file_name = './raw_data/annotations.json'):

    '''
    import class_to_category.json from gc strorage bucket
    either download it to package or not

    /!\ the raw_data is git ignored. Feel free to modify the path or git ignore file
    '''

    #print(Fore.BLUE + "Getting the model from the google storage bucket")

    BUCKET_NAME = "foodygs"
    blob_name = "foodyai_data/Training_2/annotations.json"

    storage_client = storage.Client()

    bucket = storage_client.bucket(BUCKET_NAME)

    blob = bucket.blob(blob_name)

    if download_to_disk == True:

        blob.download_to_filename(destination_file_name)
        print(Fore.BLUE +
            "Downloaded storage object {} from bucket {} to local file {}.".format(
            blob_name, BUCKET_NAME, destination_file_name)
        )

    annot = ''

    if download_to_disk == False:
        print(Fore.BLUE + f"retrieving {blob_name} from gcloud")
        annot = blob.download_as_string()
        print(Fore.GREEN + f"received... test")

    #print(Fore.GREEN + "Model retrieve, good to go")

    return annot
