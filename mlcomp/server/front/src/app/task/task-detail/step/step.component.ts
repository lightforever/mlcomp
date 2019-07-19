import {Component, OnDestroy, OnInit} from '@angular/core';
import {Location} from "@angular/common";
import {MatTreeFlatDataSource, MatTreeFlattener} from "@angular/material";
import {StepNode, FlatNode} from "../../../models";
import {FlatTreeControl} from "@angular/cdk/tree";
import {AppSettings} from "../../../app-settings";
import {of as ofObservable} from 'rxjs';
import {TaskService} from "../../task.service";

@Component({
    selector: 'app-step',
    templateUrl: './step.component.html',
    styleUrls: ['./step.component.css']
})
export class StepComponent implements OnInit, OnDestroy {
    public task: number;
    private interval: number;
    flat_node_map: Map<StepNode, FlatNode> = new Map<StepNode, FlatNode>();

    constructor(protected service: TaskService,
                protected location: Location
    ) {
    }

    private transformer = (node: StepNode, level: number) => {
        let res = {
            expandable: !!node.children && node.children.length > 0,
            name: node.name,
            level: level,
            content: node
        };
        if (node.id in this.flat_node_map) {
            let node_flat = this.flat_node_map[node.id];
            for (let k in res) {
                if(res[k]!=node_flat[k]){
                    Object.defineProperty(node_flat, k,
                        {'value': res[k]});
                }
            }

            return node_flat;
        }
        this.flat_node_map[node.id] = res;
        return res;
    };

    get_children = node => ofObservable(node.children);

    treeControl = new FlatTreeControl<FlatNode>(
        node => node.level, node => node.expandable);

    treeFlattener = new MatTreeFlattener(
        this.transformer,
            node => node.level,
            node => node.expandable,
        this.get_children);

    dataSource = new MatTreeFlatDataSource(this.treeControl,
        this.treeFlattener);

    load() {
        let self = this;
        this.service.steps(this.task).subscribe(res => {
            self.dataSource.data = res.data;
            self.treeControl.expandAll();

        });
    }

    ngOnInit() {
        this.load();
        this.interval = setInterval(() => this.load(), 3000);
    }


    node_click(node: FlatNode) {

    }

    hasChild = (_: number, node: FlatNode) => node.expandable;

    status_color(status: string) {
        switch (status) {
            case 'in_progress':
                return 'green';
            case 'failed':
                return 'red';
            case 'stopped':
                return 'orange';
            case 'successed':
                return 'green';
            default:
                throw new TypeError("unknown status: " + status)
        }

    }

    color_for_log_status(name: string, count: number) {
        return count > 0 ? AppSettings.log_colors[name] : 'gainsboro'
    }

    status_click(node: any, status: string) {
        node.content.init_level = status;
    }

    ngOnDestroy() {
        clearInterval(this.interval);
    }
}


