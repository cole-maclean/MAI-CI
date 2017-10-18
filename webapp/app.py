import flask
from flask import request
from Recommender import Recommender
from BuildRecommender import BuildRecommender
from bokeh.plotting import figure, show, output_notebook,output_file
from bokeh.models import ColumnDataSource, LabelSet
from bokeh.models import HoverTool
from bokeh.embed import components
import random
import networkx as nx
from networkx.readwrite import json_graph

app = flask.Flask(__name__)

rec = Recommender()

sc2_rec = BuildRecommender()
races = ['Terran0','Terran1']
tree = nx.DiGraph()
edge_nodes = [1,1]

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
    edge_nodes[1] = 1
    tree.add_node(1,name=first_unit,parent='null')
    return tree

@app.route("/sc2")
def index():
    default_race = 'Terran'
    races[0] = default_race + '0'
    races[1] = default_race + '1'
    default_tree(tree,races[0])
    tree_data = json_graph.tree_data(tree,root=1)
    return flask.render_template("sc2.html",tree_data=tree_data)

@app.route("/sc2_default",methods=['GET','POST'])
def default():
    tree.clear()
    races[0] = request.form['friendly_race'] + '0'
    races[1] = request.form['enemy_race'] + '1'
    default_tree(tree,races[0])
    tree_data = json_graph.tree_data(tree,root=1)
    return flask.render_template("sc2.html",tree_data=tree_data)

@app.route("/sc2_recommend",methods=['GET','POST'])
def recommend():   
    if 'autobuild' in request.form.keys():
        autobuild_len = int(request.form["length_autobuild"])
        node_order = nx.shortest_path(tree,source=edge_nodes[0],target=edge_nodes[1])     
        build_order = [tree.node[nd]['name'] for nd in node_order]
        builds = rec.predict_build(pred_input=build_order,build_length=autobuild_len,races=races)
        for bld in builds:
            next_node = len(node_order) + 1
            tree.add_node(next_node,name=bld,parent=node_order[-1])
            tree.add_edge(node_order[-1],next_node)
            node_order.append(next_node)
            edge_nodes[1] = next_node   
    else:
        expansion_unit = request.args.get('unit_id')
        custom_build = request.args.get('cust_build')
        if custom_build != 'Custom Build':
            builds = custom_build.split(',')
            node_order = nx.shortest_path(tree,source=edge_nodes[0],target=edge_nodes[1])
            build_order = [tree.node[nd]['name'] for nd in node_order]
            nd_index = build_order.index(expansion_unit)
            expansion_node = node_order[nd_index]
            tree.node[expansion_node]["name"] = builds[0]
            edge_nodes[1] = expansion_node
            node_order = nx.shortest_path(tree,source=edge_nodes[0],target=edge_nodes[1])            
            for nd in tree.nodes():
                if nd not in node_order:
                    tree.remove_node(nd)
            for bld in builds[1:]:
                next_node = len(node_order) + 1
                tree.add_node(next_node,name=bld,parent=node_order[-1])
                tree.add_edge(node_order[-1],next_node)
                node_order.append(next_node)
                edge_nodes[1] = next_node 
        else:
            node_order = nx.shortest_path(tree,source=edge_nodes[0],target=edge_nodes[1])
            build_order = [tree.node[nd]['name'] for nd in node_order]
            next_build = sc2_rec.predict_build(pred_input=build_order,build_length=1,races=races)[-1]
            next_node = len(node_order) + 1
            tree.add_node(next_node,name=next_build,parent=node_order[-1])
            tree.add_edge(node_order[-1],next_node)
            edge_nodes[1] = next_node

    tree_data = json_graph.tree_data(tree,root=1)
    return flask.render_template("sc2.html",tree_data=tree_data)

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
    return flask.render_template("index.html",script=script, div=div)

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
    return flask.render_template("all.html",script=script, div=div)

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

    return flask.render_template("recommend.html",recommendations = recommendations,script=script, div=div, flag=flag)

if __name__ == "__main__":
    import os

    port = 8000

    # Open a web browser pointing at the app.
    #os.system("open http://localhost:{0}/".format(port))

    # Set up the development server on port 8000.
    app.debug = False
    app.run()



