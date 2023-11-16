from app import db, login
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin

@login.user_loader
def load_user(id):
    return User.query.get(int(id))

followers = db.Table('followers',
    db.Column('follower_id', db.Integer, db.ForeignKey('user.id')),
    db.Column('followed_id', db.Integer, db.ForeignKey('user.id'))
)

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True)
    email = db.Column(db.String(120), index=True, unique=True)
    password_hash = db.Column(db.String(128))
    climbs = db.relationship('Climb', backref='climber', lazy='dynamic')
    about_me = db.Column(db.String(140))
    last_seen = db.Column(db.DateTime, default=datetime.utcnow)
    followed = db.relationship(
        'User', secondary=followers,
        primaryjoin=(followers.c.follower_id == id),
        secondaryjoin=(followers.c.followed_id == id),
        backref=db.backref('followers', lazy='dynamic'), lazy='dynamic')

    def __repr__(self):
        return '<User {}>'.format(self.username)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def get_climbs(self):
        climbs = Climb.query.filter_by(user_id=self.id)
        return climbs
    
    def follow(self, user):
        if not self.is_following(user):
            self.followed.append(user)

    def unfollow(self, user):
        if self.is_following(user):
            self.followed.remove(user)

    def is_following(self, user):
        return self.followed.filter(
            followers.c.followed_id == user.id).count() > 0
    
    def followed_climbs(self):
        followed = Climb.query.join(
            followers, (followers.c.followed_id == Climb.user_id)).filter(
                followers.c.follower_id == self.id, Climb.private == False)
        own = self.get_climbs()
        return followed.union(own).order_by(Climb.timestamp.desc())

class Climb(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), index=True)
    description = db.Column(db.String(512), index=True)
    colour = db.Column(db.String(64), index=True)
    grade = db.Column(db.Integer, index=True)
    timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow) 
    private = db.Column(db.Boolean)
    completed = db.Column(db.Boolean)
    # metrics
    attempt = db.Column(db.Integer)
    time_taken = db.Column(db.Float)
    av_time_bw_moves = db.Column(db.Float)
    percent_completed = db.Column(db.Float)
    difficulty = db.Column(db.String(64))
    # relationships
    screenshots = db.relationship('Screenshot', lazy='dynamic')
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', name='fk_climb_user_id'))

    def __repr__(self):
        return '<Climb {}>'.format(self.name)
    
    def get_screenshots(self):
        screenshots = Screenshot.query.filter_by(climb_id=self.id).order_by(Screenshot.order.asc())
        return screenshots
    
    def get_first_screenshot(self):
        screenshot = Screenshot.query.filter_by(climb_id=self.id).first()
        return screenshot
    
class Screenshot(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    order = db.Column(db.Integer)
    seconds_elapsed = db.Column(db.Float)
    climb_id = db.Column(db.Integer, db.ForeignKey('climb.id'))

    def __repr__(self):
        return '<Screenshot {} from Climb {}>'.format(self.id, self.climb_id)