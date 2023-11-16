pip install flask

pip install python-dotenv

pip install flask-wtf

pip install flask-sqlalchemy

pip install flask-migrate

pip install flask-moment

pip install flask-login

pip install email-validator

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``

to run the gui, navigate to the GUI folder, and then run "flask run" in the terminal

then open localhost:5000 in a browser

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

to update db by adding a column

add it to the app/models.py file first under the corresponding model

then run the following in the terminal:

flask db migrate -m "write a change message here"

flask db upgrade

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

to do: 
