from app import app, db
from flask import render_template, flash, redirect, url_for, send_from_directory, jsonify, request
from flask_login import current_user, login_user, logout_user, login_required
from app.forms import NewClimbForm, EditClimbForm, LoginForm, RegistrationForm, EditProfileForm, EmptyForm
from app.models import Climb, Screenshot, User
import os, shutil, requests
import subprocess
from werkzeug.utils import secure_filename
from werkzeug.urls import url_parse
from datetime import datetime

@app.route('/<path:filename>')
def serve_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.before_request
def before_request():
    if current_user.is_authenticated:
        current_user.last_seen = datetime.utcnow()
        db.session.commit()

@app.route('/')
@app.route('/index')
@login_required
def index():
    page = request.args.get('page', 1, type=int)
    climbs = current_user.followed_climbs().paginate(
        page=page, per_page=app.config['POSTS_PER_PAGE'], error_out=False)
    next_url = url_for('index', page=climbs.next_num) \
        if climbs.has_next else None
    prev_url = url_for('index', page=climbs.prev_num) \
        if climbs.has_prev else None
    return render_template('index.html', title='Followed Climbs', climbs=climbs.items, next_url=next_url, prev_url=prev_url)

@app.route('/explore')
@login_required
def explore():
    page = request.args.get('page', 1, type=int)
    climbs = Climb.query.filter(Climb.private == False).order_by(Climb.timestamp.desc()).paginate(
        page=page, per_page=app.config['POSTS_PER_PAGE'], error_out=False)
    next_url = url_for('explore', page=climbs.next_num) \
        if climbs.has_next else None
    prev_url = url_for('explore', page=climbs.prev_num) \
        if climbs.has_prev else None
    return render_template('index.html', title=f"Explore Public Climbs", climbs=climbs.items, next_url=next_url, prev_url=prev_url)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user is None or not user.check_password(form.password.data):
            flash('Invalid username or password')
            return redirect(url_for('login'))
        login_user(user, remember=form.remember_me.data)
        next_page = request.args.get('next')
        if not next_page or url_parse(next_page).netloc != '':
            next_page = url_for('index')
        return redirect(next_page)
    return render_template('login.html', title='Sign In', form=form)

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Congratulations, you are now a registered user!')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)

@app.route('/user/<u>')
@login_required
def user(u):
    form = EmptyForm()
    page = request.args.get('page', 1, type=int)
    user = User.query.filter(User.username == u).one()
    climbs = user.get_climbs().order_by(Climb.timestamp.desc()).paginate(
        page=page, per_page=app.config['POSTS_PER_PAGE'], error_out=False)
    next_url = url_for('user', u=user.username, page=climbs.next_num) \
        if climbs.has_next else None
    prev_url = url_for('user', u=user.username, page=climbs.prev_num) \
        if climbs.has_prev else None
    return render_template('user.html', title=f"{u}'s Climbs", climbs=climbs.items, user=user, form=form, next_url=next_url, prev_url=prev_url)

@app.route('/edit_profile', methods=['GET', 'POST'])
@login_required
def edit_profile():
    form = EditProfileForm(current_user.username)
    if form.validate_on_submit():
        current_user.username = form.username.data
        current_user.about_me = form.about_me.data
        db.session.commit()
        flash('Your changes have been saved.')
        return redirect(url_for('user', u=current_user.username))
    elif request.method == 'GET':
        form.username.data = current_user.username
        form.about_me.data = current_user.about_me
    return render_template('edit_profile.html', title='Edit Profile',
                           form=form)

@app.route('/new_climb', methods=['GET', 'POST'])
@login_required
def new_climb():
    form = NewClimbForm()
    if form.validate_on_submit():
        flash('New Climb {} added, grade {}'.format(form.name.data, form.grade.data))
        video = form.video.data
        filename = secure_filename(video.filename)
        video.save(os.path.join(app.instance_path, 'videos', filename))
        climb = Climb(name=form.name.data, description=form.description.data, 
                      grade=form.grade.data, colour=form.colour.data, attempt=form.attempt.data,
                      completed=form.completed.data, private=form.private.data, user_id=current_user.id)
        db.session.add(climb)
        db.session.commit()
        return redirect(url_for('loading', filename=filename, climb_id=climb.id)) 
    return render_template('new_climb.html', title='New Climb', form=form)

@app.route('/climb/<n>')
def climb(n):
    climb = Climb.query.filter(Climb.id == n).one()
    screenshots = climb.get_screenshots()
    return render_template('carousel.html', title=f'Climb {n}', climb=climb, screenshots=screenshots)

@app.route('/edit_climb/<n>', methods=['GET', 'POST'])
@login_required
def edit_climb(n):
    climb = Climb.query.filter(Climb.id == n).one()
    if climb.user_id != current_user.id:
        flash('You do not have permission to edit this climb')
        return redirect(url_for("index"))
    screenshots = climb.get_screenshots()
    form = EditClimbForm()
    if form.validate_on_submit():
        if form.submit.data:
            # Editing the climb data
            flash('Edited Climb {}'.format(form.name.data))
            climb.name = form.name.data
            climb.description = form.description.data
            climb.grade = form.grade.data
            climb.attempt = form.attempt.data
            climb.completed = form.completed.data
            climb.private = form.private.data
            db.session.commit()
            return redirect(url_for('climb', n=climb.id)) 
        elif form.delete.data:
            # Deleting the climb
            flash('Deleted Climb {}'.format(form.name.data))
            Screenshot.query.filter_by(climb_id=climb.id).delete()
            try:
                shutil.rmtree(f"{app.config['UPLOAD_FOLDER']}\climb{climb.id}")
            except:
                pass
            db.session.delete(climb)
            db.session.commit()
            return redirect(url_for('index'))
        elif form.delete_screenshots.data:
            # Deleting selected screenshots
            data = dict((key, request.form.getlist(key) 
                         if len(request.form.getlist(key)) > 1 
                         else request.form.getlist(key)[0]) for key in request.form.keys())
            selected_screenshot_ids=[]
            for key, value in data.items():
                if value == "on" and key.isdigit():
                    selected_screenshot_ids.append(int(key))

            order = 0
            seconds = 0
            for screenshot in screenshots:
                if screenshot.id in selected_screenshot_ids:
                    # Delete this screenshot
                    seconds += screenshot.seconds_elapsed
                    os.remove(f"{app.config['UPLOAD_FOLDER']}/climb{climb.id}/{screenshot.order}.jpg")
                    db.session.delete(screenshot)
                else:
                    # Keep this screenshot
                    if screenshot.order != order:
                        os.rename(f"{app.config['UPLOAD_FOLDER']}/climb{climb.id}/{screenshot.order}.jpg", f"{app.config['UPLOAD_FOLDER']}/climb{climb.id}/{order}.jpg")
                        screenshot.order = order
                    if order == 0:
                        screenshot.seconds_elapsed = 0
                    else:
                        screenshot.seconds_elapsed = round(screenshot.seconds_elapsed + seconds, 2)
                    seconds = 0
                    order += 1
            db.session.commit()
            return redirect(url_for('climb', n=climb.id)) 
    elif request.method == 'GET':
        form.name.data = climb.name
        form.description.data = climb.description
        form.grade.data = climb.grade
        form.completed.data = climb.completed
        form.attempt.data = climb.attempt
        form.private.data = climb.private
    return render_template('edit_climb.html', title=f'Edit Climb {n}', climb=climb, screenshots=screenshots, form=form)

@app.route('/loading/<filename>/<climb_id>')
@login_required
def loading(filename, climb_id):
    return render_template('loading.html', title='Processing New Climb', filename=filename, climb_id=climb_id)

@app.route('/run_test')
def run_test():
    arg = request.args.get('arg')
    filename = request.args.get('filename')
    climb_id = request.args.get('climb_id')
    subprocess.call(['python', '../test.py', arg, filename, climb_id])
    return jsonify({'success': True})

@app.route('/process_climb')
@login_required
def process_climb():
    filename = request.args.get('filename')
    climb_id = request.args.get('climb_id')
    endpoint = 'http://localhost:5001/receive_data'
    params = {"name": filename, "climb_id": climb_id}
    response = requests.get(url=endpoint, params=params)
    #subprocess.call(['python', '../video_processing.py', filename, climb_id])
    return jsonify({'success': True})

@app.route('/delete_file')
@login_required
def delete_file():
    filename = request.args.get('filename')
    os.remove(app.config['UPLOAD_FOLDER']+f"\\videos\\{filename}")
    return jsonify({'success': True})

@app.route('/about')
def about():
    return render_template('about.html', title='About')

@app.route('/follow/<username>', methods=['POST'])
@login_required
def follow(username):
    form = EmptyForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=username).first()
        if user is None:
            flash('User {} not found.'.format(username))
            return redirect(url_for('index'))
        if user == current_user:
            flash('You cannot follow yourself!')
            return redirect(url_for('user', u=username))
        current_user.follow(user)
        db.session.commit()
        flash('You are following {}!'.format(username))
        return redirect(url_for('user', u=username))
    else:
        return redirect(url_for('index'))

@app.route('/unfollow/<username>', methods=['POST'])
@login_required
def unfollow(username):
    form = EmptyForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=username).first()
        if user is None:
            flash('User {} not found.'.format(username))
            return redirect(url_for('index'))
        if user == current_user:
            flash('You cannot unfollow yourself!')
            return redirect(url_for('user', u=username))
        current_user.unfollow(user)
        db.session.commit()
        flash('You are not following {}.'.format(username))
        return redirect(url_for('user', u=username))
    else:
        return redirect(url_for('index'))