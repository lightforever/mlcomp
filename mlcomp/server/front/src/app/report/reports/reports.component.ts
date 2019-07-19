import {Component} from '@angular/core';
import {Paginator} from "../../paginator";
import {Location} from "@angular/common";
import {Report, ReportsFilter} from "../../models";
import {ReportService} from "../report.service";
import {MatDialog} from "@angular/material";
import {ReportAddDialogComponent} from "./report-add-dialog";

@Component({
    selector: 'app-reports',
    templateUrl: './reports.component.html',
    styleUrls: ['./reports.component.css']
})
export class ReportsComponent extends Paginator<Report> {

    task: number;
    dag: number;

    protected displayed_columns: string[] = [
        "id",
        "name",
        "time",
        "tasks",
        "tasks_not_finished"
    ];

    constructor(
        protected service: ReportService,
        protected location: Location,
        public dialog: MatDialog
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


    add() {
        this.service.add_start().subscribe(res => {
            const dialogRef = this.dialog.open(ReportAddDialogComponent,
                {
                    width: '600px', height: '300px',
                    data: res
                });

            dialogRef.afterClosed().subscribe(result => {
                if (result) {
                    this.service.add_end(result).subscribe(_ => {
                        this.change.emit();
                    });

                }
            });
        });

    }
}
