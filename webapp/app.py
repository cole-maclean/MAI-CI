import flask
from flask import request
from Recommender import Recommender
from bokeh.plotting import figure, show, output_notebook,output_file
from bokeh.models import ColumnDataSource, LabelSet
from bokeh.models import HoverTool
from bokeh.embed import components
import random


app = flask.Flask(__name__)

rec = Recommender()



@app.route("/")
def index():   
    sparse_labels = [lbl if random.random() <=0.01 else '' for lbl in rec.labels]
    source = ColumnDataSource({'x':rec.embedding_weights[:,0],'y':rec.embedding_weights[:,1],'labels':rec.labels,'sparse_labels':sparse_labels})
    hover = HoverTool(
            tooltips="""
            <div>
                <span>@labels</span>
            </div>
            """
        )

    TOOLS=[hover,"crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,"]

    p = figure(tools=TOOLS)

    p.scatter("x", "y", radius=0.1, fill_alpha=0.6,
              fill_color="#c7e9b4",
              line_color=None,source=source)

    labels = LabelSet(x="x", y="y", text="sparse_labels", y_offset=8,
                      text_font_size="8pt", text_color="#555555", text_align='center',
                     source=source)
    p.add_layout(labels)
    script, div = components(p)
    return flask.render_template("index.html",script=script, div=div)

@app.route("/recommend")
def recommend():
    #initial call to runplan which displays the planner simulation
    user = request.args.get('user')
    recommendations = rec.user_recs(user)
    colors = []
    for label in rec.labels:
        if label in recommendations:
            colors.append("#0c2c84")
        elif label in rec.user_subs:
            colors.append("#7fcdbb")
        else:
            colors.append("#c7e9b4")

    sparse_labels = [lbl if lbl in recommendations or lbl in rec.user_subs else '' for lbl in rec.labels]
    source = ColumnDataSource({'x':rec.embedding_weights[:,0],'y':rec.embedding_weights[:,1],'labels':rec.labels,'sparse_labels':sparse_labels,'colors':colors})
    hover = HoverTool(
            tooltips="""
            <div>
                <span>@labels</span>
            </div>
            """
        )

    TOOLS=[hover,"crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,"]

    p = figure(tools=TOOLS)

    p.scatter("x", "y", radius=0.1, fill_alpha=0.6,
              fill_color="colors",
              line_color=None,source=source)

    labels = LabelSet(x="x", y="y", text="sparse_labels", y_offset=8,
                      text_font_size="8pt", text_color="#555555", text_align='center',
                     source=source)
    p.add_layout(labels)
    script, div = components(p)

    return flask.render_template("recommend.html",recommendations = recommendations,script=script, div=div)

if __name__ == "__main__":
    import os

    port = 8000

    # Open a web browser pointing at the app.
    #os.system("open http://localhost:{0}/".format(port))

    # Set up the development server on port 8000.
    app.debug = True
    app.run()



