import {Component, OnInit} from '@angular/core';
import {Paginator} from "../../paginator";
import {TasksComponent} from "../../task/tasks/tasks.component";
import {ReportService} from "../../report.service";
import {Location} from "@angular/common";
import {ActivatedRoute, Router} from "@angular/router";
import {MatIconRegistry} from "@angular/material";
import {DomSanitizer} from "@angular/platform-browser";
import {MessageService} from "../../message.service";
import {DagFilter, ReportsFilter} from "../../models";

@Component({
    selector: 'app-reports',
    templateUrl: './reports.component.html',
    styleUrls: ['./reports.component.css']
})
export class ReportsComponent extends Paginator<TasksComponent> {

    task: number;
    dag: number;

    constructor(protected service: ReportService, protected location: Location,
                private router: Router, private  route: ActivatedRoute,
                iconRegistry: MatIconRegistry, sanitizer: DomSanitizer,
                private message_service: MessageService
    ) {
        super(service, location);
    }

    protected _ngOnInit() {
        let url = this.router.url;
        if (url.indexOf('task-detail') != -1) {
            this.task = parseInt(this.route.snapshot.paramMap.get('id'));
        }
        else if(url.indexOf('dag-detail')!=-1){
            this.dag = parseInt(this.route.snapshot.paramMap.get('id'));
        }
    }

     get_filter(): any {
        let res = new ReportsFilter();
        res.paginator = super.get_filter();
        res.task = this.task;
        res.dag = this.dag;
        return res;
    }

    protected displayed_columns: string[] = ["id", "name", "time", "tasks", "tasks_not_finished"];

}
