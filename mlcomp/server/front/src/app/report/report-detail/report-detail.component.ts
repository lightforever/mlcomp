import {Component, OnInit} from '@angular/core';
import {Location} from "@angular/common";
import {ActivatedRoute, Router} from "@angular/router";
import {Report} from "../../models";
import {DynamicresourceService} from "../../dynamicresource.service";
import {ReportService} from "../report.service";
import {MatDialog} from "@angular/material";
import {ReportUpdateDialogComponent} from "./report-update-dialog.component";

@Component({
    selector: 'app-report-detail',
    templateUrl: './report-detail.component.html',
    styleUrls: ['./report-detail.component.css']
})
export class ReportDetailComponent implements OnInit {
    id: number;
    report: Report;
    dag_panel_open: boolean = false;
    task_panel_open: boolean = false;

    constructor(protected service: ReportService,
                protected location: Location,
                private router: Router,
                private  route: ActivatedRoute,
                protected resource_service: DynamicresourceService,
                public update_dialog: MatDialog
    ) {
    }

    ngOnInit() {
        this.id = parseInt(this.route.snapshot.paramMap.get('id'));
        this.resource_service.load('plotly').then(() => {
            this.service.get_obj<Report>(this.id).subscribe(data => {
                this.report = data;
            });
        });
    }

    update_layout() {
        this.service.update_layout_start(this.id).subscribe(data => {
            let config = {
                width: '250px', height: '180px',
                data: data
            };

            let dialog = this.update_dialog.open(
                ReportUpdateDialogComponent,
                config);
            dialog.afterClosed().subscribe(res => {
                if(!res||!res.id){
                    return;
                }

                this.service.update_layout_end(res.id, res.layout).
                subscribe(res => {
                    this.report = res;
                });
            });
        });


    }
}
