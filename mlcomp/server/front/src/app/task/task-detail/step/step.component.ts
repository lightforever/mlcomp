import { Component, OnInit } from '@angular/core';
import {Location} from "@angular/common";
import {ActivatedRoute, Router} from "@angular/router";
import {MatIconRegistry, MatTreeFlatDataSource, MatTreeFlattener} from "@angular/material";
import {DomSanitizer} from "@angular/platform-browser";
import {MessageService} from "../../../message.service";
import {StepNode, FlatNode} from "../../../models";
import {FlatTreeControl} from "@angular/cdk/tree";
import {TaskService} from "../../../task.service";
import {AppSettings} from "../../../app-settings";

@Component({
  selector: 'app-step',
  templateUrl: './step.component.html',
  styleUrls: ['./step.component.css']
})
export class StepComponent implements OnInit {
  private step_id: number;

 constructor(protected service: TaskService, protected location: Location,
                private router: Router, private  route: ActivatedRoute,
                iconRegistry: MatIconRegistry, sanitizer: DomSanitizer,
                private message_service: MessageService
    ){}

  private transformer = (node: StepNode, level: number) => {
        return {
            expandable: !!node.children && node.children.length > 0,
            name: node.name,
            level: level,
            content: node
        };
    };

    treeControl = new FlatTreeControl<FlatNode>(
        node => node.level, node => node.expandable);

    treeFlattener = new MatTreeFlattener(
        this.transformer, node => node.level, node => node.expandable, node => node.children);

    dataSource = new MatTreeFlatDataSource(this.treeControl, this.treeFlattener);

    ngOnInit() {
        let self = this;
        this.step_id = parseInt(this.route.parent.snapshot.paramMap.get('id'));
        this.service.steps(this.step_id).subscribe(res => {
            self.dataSource.data = res;
            self.treeControl.expandAll();
        });
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
        throw new TypeError("unknown status: "+status)
    }

  }

  color_for_log_status(name: string, count: number) {
      return count > 0 ? AppSettings.log_colors[name] : 'gainsboro'
  }

  status_click(node: any, status: string) {
     node.content.init_level = status;
  }
}


