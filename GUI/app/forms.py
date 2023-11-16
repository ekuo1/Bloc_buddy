from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, IntegerField, SubmitField, SelectField, PasswordField, BooleanField
from wtforms.validators import InputRequired, Optional, NumberRange, Length, NoneOf, Email, EqualTo, ValidationError
from flask_wtf.file import FileField, FileRequired, FileAllowed
from app.models import User

colour_choices = ["Select", "red", "green", "blue", "yellow", "pink", "purple", "black"]

class NewClimbForm(FlaskForm):
    name = StringField('Name*', validators=[InputRequired(), Length(max=128)])
    description = TextAreaField('Description', validators=[Length(max=512)])
    grade = IntegerField('Grade', validators=[NumberRange(min=0, max=9), Optional()])
    colour = SelectField('Colour*', choices=colour_choices, validators=[InputRequired(), NoneOf(["Select"])])
    attempt = IntegerField('Number of Attempts', validators=[Optional(), NumberRange(min=1, message='Must enter a number greater than 0')])
    completed = BooleanField('Completed Climb', default=True)
    private = BooleanField('Private Climb')
    video = FileField('Video*', validators=[FileRequired(), FileAllowed(['mp4'], '.mp4 videos only!')])
    submit = SubmitField('Submit', render_kw={'class': 'btn btn-primary btn-sm'})

class EditClimbForm(FlaskForm):
    name = StringField('Name*', validators=[InputRequired(), Length(max=128)])
    description = TextAreaField('Description', validators=[Length(max=512)])
    grade = IntegerField('Grade', validators=[NumberRange(min=0, max=9), Optional()])
    attempt = IntegerField('Number of Attempts', validators=[Optional(), NumberRange(min=1, message='Must enter a number greater than 0')])
    completed = BooleanField('Completed Climb', default=True)
    private = BooleanField('Private Climb')
    submit = SubmitField('Submit', render_kw={'class': 'btn btn-primary btn-sm'})
    delete = SubmitField('Delete Climb', render_kw={'class':'btn btn-danger btn-sm', 'onclick': "return confirm('Are you sure you want to delete this climb? This cannot be undone.');"})
    delete_screenshots = SubmitField("Delete Selected Screenshots", render_kw={'class':'btn btn-danger btn-sm', 'onclick': "return confirm('Are you sure you want to delete these screenshots? This cannot be undone.');"})

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired()])
    password = PasswordField('Password', validators=[InputRequired()])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Sign In', render_kw={'class': 'btn btn-primary btn-sm'})

class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired()])
    email = StringField('Email', validators=[InputRequired(), Email()])
    password = PasswordField('Password', validators=[InputRequired()])
    password2 = PasswordField(
        'Repeat Password', validators=[InputRequired(), EqualTo('password')])
    submit = SubmitField('Register', render_kw={'class': 'btn btn-primary btn-sm'})

    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user is not None:
            raise ValidationError('Please use a different username.')

    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user is not None:
            raise ValidationError('Please use a different email address.')
        
class EditProfileForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired()])
    about_me = TextAreaField('About me', validators=[Length(min=0, max=140)])
    submit = SubmitField('Submit', render_kw={'class': 'btn btn-primary btn-sm'})

    def __init__(self, original_username, *args, **kwargs):
        super(EditProfileForm, self).__init__(*args, **kwargs)
        self.original_username = original_username

    def validate_username(self, username):
        if username.data != self.original_username:
            user = User.query.filter_by(username=self.username.data).first()
            if user is not None:
                raise ValidationError('Please use a different username.')
            
class EmptyForm(FlaskForm):
    submit = SubmitField('Submit', render_kw={'class': 'btn btn-primary btn-sm'})