import {Component, OnInit} from '@angular/core';
import {ReportService} from "../../report.service";
import {Location} from "@angular/common";
import {ActivatedRoute, Router} from "@angular/router";
import {Report} from "../../models";
import {DynamicresourceService} from "../../dynamicresource.service";

@Component({
    selector: 'app-report-detail',
    templateUrl: './report-detail.component.html',
    styleUrls: ['./report-detail.component.css']
})
export class ReportDetailComponent implements OnInit {
    id: string;
    report: Report;
    dag_panel_open: boolean = false;
    task_panel_open: boolean = false;

    constructor(protected service: ReportService, protected location: Location,
                private router: Router, private  route: ActivatedRoute,
                protected resource_service: DynamicresourceService
    ) {
    }

    load(){
        this.service.get_obj<Report>(this.id).subscribe(data => {
                this.report = data;
            });
    }

    ngOnInit() {
        let self = this;
        this.id = this.route.snapshot.paramMap.get('id');
        this.service.data_updated.subscribe(res=>{
            self.load();
        });
        this.resource_service.load('plotly').then(() => {
            self.load();
        });
    }

}
