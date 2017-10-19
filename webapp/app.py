from flask import Flask, request, render_template, make_response
from Recommender import Recommender
from BuildRecommender import BuildRecommender
from bokeh.plotting import figure, show, output_notebook,output_file
from bokeh.models import ColumnDataSource, LabelSet
from bokeh.models import HoverTool
from bokeh.embed import components
import random
import networkx as nx
from networkx.readwrite import json_graph
import json

app = Flask(__name__)

rec = Recommender()

sc2_rec = BuildRecommender()

def race_first_unit(race):
    if race == 'Terran0':
        first_unit = 'CommandCenter0'
    elif race == 'Protoss0':
        first_unit = 'Nexus0'
    elif race =='Zerg0':
        first_unit = 'Hatchery0'
    return first_unit

def default_tree(tree,race):
    first_unit = race_first_unit(race)
    tree.add_node(first_unit,name=first_unit,parent='null')
    return tree, first_unit

def build_response(tree,races,root):
    tree_data = json_graph.tree_data(tree,root=root)
    resp = make_response(render_template('sc2.html',tree_data=tree_data,race=races[0],enemy_race=races[1]))
    resp.set_cookie('tree_data', json.dumps(tree_data))
    resp.set_cookie('races',json.dumps(races))
    print(json.dumps(tree_data))
    return resp

@app.route("/sc2")
def sc2_index():
    tree = nx.DiGraph()
    default_race = 'Terran'
    races = [default_race + '0', default_race + '1']
    tree, root = default_tree(tree,races[0])
    resp = build_response(tree,races,root)
    return resp

@app.route("/sc2_default",methods=['GET','POST'])
def sc2_default():
    tree= nx.DiGraph()
    races = [request.form['friendly_race'] + '0',
             request.form['enemy_race'] + '1']
    tree,root = default_tree(tree,races[0])
    resp = build_response(tree,races,root)
    return resp

@app.route("/sc2_recommend",methods=['GET','POST'])
def sc2_recommend():
    tree_data = json.loads(request.cookies.get('tree_data'))
    print(tree_data)
    races = json.loads(request.cookies.get('races'))
    tree = json_graph.tree_graph(tree_data,{'id':'id','children':'children','parent':'parent','name':'name'})
    root = race_first_unit(races[0])
    if 'autobuild' in request.form.keys():
        autobuild_len = int(request.form["length_autobuild"])
        leaf = [n for n,d in tree.out_degree().items() if d==0][0]    
        build_order = nx.shortest_path(tree,source=root,target=leaf)
        builds = sc2_rec.predict_build(pred_input=build_order,build_length=autobuild_len,races=races)
        for bld in builds:
            tree.add_node(bld,name=bld,parent=build_order[-1])
            tree.add_edge(build_order[-1],bld)
            build_order.append(bld) 
    else:
        expansion_unit = request.args.get('unit_id')
        custom_build = request.args.get('cust_build')
        if custom_build != 'Custom Build':
            builds = custom_build.split(',')
            build_order = nx.shortest_path(tree,source=root,target=expansion_unit)[:-1]       
            for nd in tree.nodes():
                if nd not in build_order:
                    tree.remove_node(nd)
            for bld in builds:
                tree.add_node(bld,name=bld,parent=build_order[-1])
                tree.add_edge(build_order[-1],bld)
                build_order.append(bld)
        else:
            expansion_unit = request.args.get('unit_id')
            build_order = nx.shortest_path(tree,source=root,target=expansion_unit)
            for nd in tree.nodes():
                if nd not in build_order:
                    tree.remove_node(nd)
            next_build = sc2_rec.predict_build(pred_input=build_order,build_length=1,races=races)[-1]
            tree.add_node(next_build,name=next_build,parent=build_order[-1])
            tree.add_edge(build_order[-1],next_build)

    resp = build_response(tree,races,root)

    return resp

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
    p.sizing_mode = 'scale_width'

    p.scatter("x", "y", radius=0.1, fill_alpha=0.6,
              fill_color="#c7e9b4",
              line_color=None,source=source)

    labels = LabelSet(x="x", y="y", text="sparse_labels", y_offset=8,
                      text_font_size="8pt", text_color="#555555", text_align='center',
                     source=source)
    p.add_layout(labels)
    script, div = components(p)
    return render_template("index.html",script=script, div=div)

@app.route("/all")
def all():   
    source = ColumnDataSource({'x':rec.embedding_weights[:,0],'y':rec.embedding_weights[:,1],'labels':rec.labels,'sparse_labels':rec.labels})
    hover = HoverTool(
            tooltips="""
            <div>
                <span>@labels</span>
            </div>
            """
        )

    TOOLS=[hover,"crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,"]

    p = figure(tools=TOOLS)
    p.sizing_mode = 'scale_width'

    p.scatter("x", "y", radius=0.1, fill_alpha=0.6,
              fill_color="#c7e9b4",
              line_color=None,source=source)

    labels = LabelSet(x="x", y="y", text="sparse_labels", y_offset=8,
                      text_font_size="8pt", text_color="#555555", text_align='center',
                     source=source)
    p.add_layout(labels)
    script, div = components(p)
    return render_template("all.html",script=script, div=div)

@app.route("/recommend")
def recommend():
    #initial call to runplan which displays the planner simulation
    user = request.args.get('user')
    if len(user) <=1:
        return 'Please input a username'
    recommendations = rec.user_recs(user)
    flag = ''
    if recommendations == []:
        flag = "Not enough comment history to provide recommendations"
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
    p.sizing_mode = 'scale_width'

    p.scatter("x", "y", radius=0.1, fill_alpha=0.6,
              fill_color="colors",
              line_color=None,source=source)

    labels = LabelSet(x="x", y="y", text="sparse_labels", y_offset=8,
                      text_font_size="8pt", text_color="#555555", text_align='center',
                     source=source)
    p.add_layout(labels)
    script, div = components(p)

    return render_template("recommend.html",recommendations = recommendations,script=script, div=div, flag=flag)

if __name__ == "__main__":
    import os

    port = 8000

    # Open a web browser pointing at the app.
    #os.system("open http://localhost:{0}/".format(port))

    # Set up the development server on port 8000.
    app.debug = False
    app.run()



