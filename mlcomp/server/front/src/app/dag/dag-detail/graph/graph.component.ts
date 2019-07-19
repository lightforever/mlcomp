import {AfterViewInit, Component, OnDestroy, OnInit} from '@angular/core';
import {MessageService} from "../../../message.service";
import {ActivatedRoute, Router} from "@angular/router";
import {DagDetailService} from "../dag-detail/dag-detail.service";
import {DynamicresourceService} from "../../../dynamicresource.service";
import {AppSettings} from "../../../app-settings";

@Component({
    selector: 'app-graph',
    templateUrl: './graph.component.html',
    styleUrls: ['./graph.component.css']
})
export class GraphComponent implements AfterViewInit, OnDestroy {

    public dag: number;
    interval: number;
    data;

    constructor(private message_service: MessageService, private route: ActivatedRoute,
                private service: DagDetailService,
                private resource_service: DynamicresourceService,
                private router: Router
    ) {
    }

    ngAfterViewInit() {
        this.load_network();
        this.interval = setInterval(() => this.load_network(), 3000);
    }

    private load_network() {
        let self = this;
        this.resource_service.load('vis.min.js', 'vis.min.css').then(res => {
            this.service.get_graph(this.dag).subscribe(res => {
                res.nodes.forEach(obj => obj.color = AppSettings.status_colors[obj.status]);
                // res.nodes.forEach(obj => obj.color = 'green');
                res.edges.forEach(obj => obj.color = AppSettings.status_colors[obj.status]);

                let vis = window['vis'];
                let nodes = new vis.DataSet(res.nodes);
                // create an array with edges
                let edges = new vis.DataSet(res.edges);

                // create a network
                let container = document.getElementById('mynetwork');
                if (!this.data) {
                    this.data = {
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

                    let network = new vis.Network(container, this.data, options);
                    network.on('doubleClick', function (properties) {
                        var ids = properties.nodes;
                        var clickedNodes = nodes.get(ids);
                        self.router.navigate(['/tasks/task-detail/' + clickedNodes[0].id + '/logs']);
                    });
                    return;
                }

                for (let item of res.nodes) {
                    this.data.nodes.update(item)
                }
                for (let item of res.edges) {
                    this.data.nodes.update(item)
                }


            });
        }).catch(err => this.message_service.add(err));


    }

    ngOnDestroy(): void {
        clearInterval(this.interval);
    }
}
