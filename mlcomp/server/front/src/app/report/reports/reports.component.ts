import {Component, OnInit} from '@angular/core';
import {Paginator} from "../../paginator";
import {TasksComponent} from "../../task/tasks/tasks.component";
import {ReportService} from "../../report.service";
import {Location} from "@angular/common";
import {DagFilter, ReportsFilter} from "../../models";

@Component({
    selector: 'app-reports',
    templateUrl: './reports.component.html',
    styleUrls: ['./reports.component.css']
})
export class ReportsComponent extends Paginator<TasksComponent> {

    task: number;
    dag: number;

    constructor(protected service: ReportService, protected location: Location
    ) {
        super(service, location);
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
