import {Component, OnInit} from '@angular/core';
import {ReportService} from "../../../report.service";
import {Location} from "@angular/common";
import {ActivatedRoute, Router} from "@angular/router";
import {ReportTile} from "../../../models";
import {DynamicresourceService} from "../../../dynamicresource.service";

@Component({
    selector: 'app-report-detail',
    templateUrl: './report-detail.component.html',
    styleUrls: ['./report-detail.component.css']
})
export class ReportDetailComponent implements OnInit {
    id: string;
    report: ReportTile[] = [];
    dag_panel_open: boolean = false;
    task_panel_open: boolean = false;

    constructor(protected service: ReportService, protected location: Location,
                private router: Router, private  route: ActivatedRoute,
                protected resource_service: DynamicresourceService
    ) {
    }

    load(){
        this.service.get_obj<ReportTile[]>(this.id).subscribe(data => {
                this.report = data;
                setTimeout(() => {
                    for (let i in data) {
                        let tile = data[i];
                        if (tile.type == 'series') {
                            let id = 'series_' + i.toString();
                            if (document.getElementById(id)) {
                                let layout = {
                                    'title': tile.name,
                                    'height': 300,
                                    'width': 600,
                                    'margin': {'b': 40, 't':40}
                                };
                                window['Plotly'].newPlot(id, tile.data, layout, {showSendToCloud: true});

                            }
                        }


                    }

                }, 100);

            });
    }

    ngOnInit() {
        let self = this;
        this.id = this.route.snapshot.paramMap.get('id');
        this.service.data_updated.subscribe(data=>{
            self.load();
        });
        this.resource_service.load('plotly').then(() => {
            self.load();
        });
    }

}
