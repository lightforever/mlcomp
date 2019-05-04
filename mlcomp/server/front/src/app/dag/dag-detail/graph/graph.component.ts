import {Component, OnInit} from '@angular/core';
import {MessageService} from "../../../message.service";
import {ActivatedRoute, Router} from "@angular/router";
import {DagDetailService} from "../../../dag-detail.service";
import {DynamicresourceService} from "../../../dynamicresource.service";
import {AppSettings} from "../../../app-settings";

@Component({
    selector: 'app-graph',
    templateUrl: './graph.component.html',
    styleUrls: ['./graph.component.css']
})
export class GraphComponent implements OnInit {

    private dag_id: string;

    constructor(private message_service: MessageService, private route: ActivatedRoute,
                private service: DagDetailService,
                private resource_service: DynamicresourceService,
                private router: Router
    ) {
    }

    ngOnInit() {
        this.dag_id = this.route.parent.snapshot.paramMap.get('id');
        this.load_network();
    }

    private load_network() {
        let self = this;
        this.resource_service.load('vis.min.js', 'vis.min.css').then(res => {
            this.service.get_graph(this.dag_id).subscribe(res => {
                res.nodes.forEach(obj => obj.color = AppSettings.status_colors[obj.status]);
                res.edges.forEach(obj => obj.color = AppSettings.status_colors[obj.status]);

                let vis = window['vis'];
                let nodes = new vis.DataSet(res.nodes);
                // create an array with edges
                let edges = new vis.DataSet(res.edges);

                // create a network
                let container = document.getElementById('mynetwork');
                let data = {
                    nodes: nodes,
                    edges: edges
                };
                let options = {
                    layout: {
                        hierarchical: {
                            direction: 'LR',
                            "sortMethod": "directed",

                        },

                    },

                    edges: {
                        arrows: 'to'
                    }
                };
                let network = new vis.Network(container, data, options);
                network.on('doubleClick', function (properties) {
                    var ids = properties.nodes;
                    var clickedNodes = nodes.get(ids);
                    self.router.navigate(['/task/'+clickedNodes[0].id]);
                });

            });
        }).catch(err => this.message_service.add(err));


    }
}
