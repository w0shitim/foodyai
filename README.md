# Foodyai project
Making it easy for everyone to track their nutrition intake.

Foodyai is the final project of a team of batch #985 - LeWagon Data Science Part-time Bootcamp!

# How it started
Food is present in our lives every single day! It is essential for our survival, but many of us, either for medical reasons or calorie intake control, have a need for tracking the nutritional composition of our diet.

How can we do it easily? By simply taking a picture of our food intake and get the nutritional information right away!

To achieve that, we can make use of models that recognize food from images.

# Data
The dataset used consists in food images collected through the MyFoodRepo app, where numerous volunteer Swiss users provide images of their daily food intake in the context of a digital cohort called Food & You. It contains a total of 54392 RGB images of food items with 100256 annotations and 323 food classes.

# Model
A pre-trained model was used for the task. The selcted model was Detectron2, Facebook AI Research's (FAIR’s) next generation library that provides state-of-the-art detection and segmentation algorithms.

The framework consists on training the model with the mentioned dataset and then feeding it with an imaage of a food item and detect the food category. Afterwards, the nutrition information of that food category is extracted through an API and the results is shown in the Foodyai web application.


# Startup the project

The initial setup.

Create virtualenv and install the project:
```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv ~/venv ; source ~/venv/bin/activate ;\
    pip install pip -U; pip install -r requirements.txt
```

Unittest test:
```bash
make clean install test
```

Check for foodyai in gitlab.com/{group}.
If your project is not set please add it:

- Create a new project on `gitlab.com/{group}/foodyai`
- Then populate it:

```bash
##   e.g. if group is "{group}" and project_name is "foodyai"
git remote add origin git@github.com:{group}/foodyai.git
git push -u origin master
git push -u origin --tags
```

Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
foodyai-run
```

# Install

Go to `https://github.com/{group}/foodyai` to see the project, manage issues,
setup you ssh public key, ...

Create a python3 virtualenv and activate it:

```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv -ppython3 ~/venv ; source ~/venv/bin/activate
```

Clone the project and install it:

```bash
git clone git@github.com:{group}/foodyai.git
cd foodyai
pip install -r requirements.txt
make clean install test                # install and test
```
Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
foodyai-run
```
