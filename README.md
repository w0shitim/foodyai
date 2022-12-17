# About

<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQGaGh1OpjZGkCh3y8lb3ty7oR3iPNsS-uNLw&usqp=CAU" alt="Le-wagon-logo" width="300"/>

<br/>
This project is part of the curriculum of the bootcamp Le Wagon, for the Data Science batch 985 (part time).
It was a 8 days project with the goal of wrapping everything we learned during the bootcamp.
We had the choice to choose whichever topic we wished.
The project was realized in a group of 4 people.

<br/>

# Foodyai project
<b>Making it easy for everyone to track their nutrition intake.</b>

<br/>

# How it started
Food is present in our lives every single day! It is essential for our survival, but many of us, either for medical reasons or calorie intake control, have a need for tracking the nutritional composition of our diet.

How can we do it easily? By simply taking a picture of our food intake and get the nutritional information right away!

To achieve that, we can make use of models that recognize food from images.

<br/>

# Data

<img src="https://play-lh.googleusercontent.com/b3cRrx46x8m5YskE63TSVLoj6lOhRbSszVTlkgK5m0jD7krL9pdeKzHLdboR8ulQYXM" alt="my-food-repo-logo" width="200"/>

<br/>
The dataset used consists in food images collected through the MyFoodRepo app, where numerous volunteer Swiss users provide images of their daily food intake in the context of a digital cohort called Food & You. It contains a total of 54392 RGB images of food items with 100256 annotations and 323 food classes.

<br/>
<img src="https://camo.githubusercontent.com/2dd0fdc7e950a8ce7cf829286c54a7aa1b7967936cd4b6b1bb8b41806fed7dfd/68747470733a2f2f692e696d6775722e636f6d2f7a53324e6266302e706e67" alt="food-image-segmented" width="200"/>

<br/>

# Model

<img src="https://curiousily.com/static/dff66fd0972574ae284f7df9533d369f/3e3fe/detectron2-logo.png" alt="detectron2-logo" width="200"/>

<br/>
A pre-trained model was used for the task. The selected model was Detectron2, Facebook AI Research's (FAIR’s) next generation library that provides state-of-the-art detection and segmentation algorithms.

<br/>
<p align="center">
<img src="https://i0.wp.com/roboticseabass.com/wp-content/uploads/2020/11/object_detection_types.png?resize=750%2C306&ssl=1" alt="detectron2-example" width="500"/>
</p>

<br/>
The framework consists on training the model with the mentioned dataset and then feeding it with an image of a food item and detect the food category. Afterwards, the nutrition information of that food category is extracted through an API and the results is shown in the Foodyai web application.

<br/>

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
