from flask import Flask, render_template, redirect, url_for, request,session,Response
from support_file import get_frame


app = Flask(__name__)

app.secret_key = '1234'


@app.route('/', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] != 'admin' or request.form['password'] != 'admin':
            error = 'Invalid Credentials. Please try again.'
        else:
            return redirect(url_for('option'))
    return render_template('login.html', error=error)

@app.route('/option',methods=['GET', 'POST'])
def option():
    if request.method == 'POST':
        if request.form['scen'] == 'College':
            print('College')
            session['scen'] = request.form['scen']
            return redirect(url_for('gesture'))

        elif request.form['scen'] == 'Home':
            print('Home')
            session['scen'] = request.form['scen']
            return redirect(url_for('gesture'))

        elif request.form['scen'] == 'Airport':
            print('Airport')
            session['scen'] = request.form['scen']
            return redirect(url_for('gesture'))

        elif request.form['scen'] == 'Office':
            print('Office')
            session['scen'] = request.form['scen']
            return redirect(url_for('gesture'))

        elif request.form['scen'] == 'Restaurant':
            print('Restaurant')
            session['scen'] = request.form['scen']
            return redirect(url_for('gesture'))

    return render_template('option.html')


@app.route('/gesture')
def gesture():
    #scen = session.get('scen',None)
    #print(scen)
    return render_template('gesture.html')

@app.route('/video_stream')
def video_stream():
     scen = session.get('scen',None)
     print(scen)
     return Response(get_frame(scen),mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, threaded=True)
