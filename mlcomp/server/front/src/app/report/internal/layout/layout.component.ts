import {Component, Input, OnDestroy, OnInit} from '@angular/core';
import {Report, ReportItem} from "../../../models";
import {ReportService} from "../../../report.service";
import {LayoutService} from "./layout.service";
import {SeriesComponent} from "../series/series.component";
import {Helpers} from "../../../helpers";

@Component({
    selector: 'app-layout',
    templateUrl: './layout.component.html',
    styleUrls: ['./layout.component.css']
})
export class LayoutComponent implements OnInit, OnDestroy {
    @Input() item: ReportItem;
    @Input() data;
    @Input() report_id: number;
    @Input() id: string;
    private interval: number;
    items_joined: ReportItem[] = [];
    items_joined_data: any[] = [];

    constructor(protected service: ReportService,
                protected layout_service: LayoutService) {
    }

    ngOnInit() {
        this.subscribe_report_changed();
        this.form_items_joined();
    }

    form_items_joined() {
        this.items_joined = [];
        if (!this.item.items) {
            return;
        }
        for (let child of this.item.items) {
            if (child.type == 'series') {
                let subchildren = SeriesComponent.create(child, this.data[child.source]);
                for (let s of subchildren) {
                    this.items_joined.push(s[0]);
                    this.items_joined_data.push(s[1]);
                }
            } else if (['img_classify', 'img'].indexOf(child.type) != -1) {
                let i = 0;
                for (let d of this.data[child.source]) {
                    let child_clone = Helpers.clone(child);
                    child_clone.index = i;
                    this.items_joined.push(child_clone);
                    this.items_joined_data.push(d);
                    i++;
                }
            } else {
                this.items_joined.push(child);
                this.items_joined_data.push(this.data);
            }
        }
    }

    td_width(child: ReportItem) {
        let total = 0;
        for (let c of this.items_joined) {
            total += c.cols ? c.cols : 1;
        }

        return (100 * (child.cols ? child.cols : 1) / total).toString() + '%';
    }

    private subscribe_report_changed() {
        if (this.report_id != null) {
            this.interval = setInterval(() => this.update(), 5000);

            this.service.data_updated.subscribe(res => {
                this.update(true);
            });
        }
    }

    private update(hard = false) {
        this.service.get_obj<Report>(this.report_id).subscribe(data => {
            if (this.item && !hard) {
                if (JSON.stringify(this.item) != JSON.stringify(data.layout)) {
                    this.data = data.data;
                    this.item = data.layout;
                    this.form_items_joined();
                } else {
                    for (let key in data.data) {
                        // noinspection JSUnfilteredForInLoop
                        if (JSON.stringify(this.data[key]) != JSON.stringify(data.data[key])) {
                            if (data.data[key].length != this.data[key].length) {
                                this.data = data.data;
                                this.item = data.layout;
                                this.form_items_joined();
                                break;
                            } else {
                                // noinspection JSUnfilteredForInLoop
                                this.layout_service.data_updated.emit({'key': key, 'data': data.data[key]});
                                // noinspection JSUnfilteredForInLoop
                                this.data[key] = data.data[key];
                            }

                        }
                    }
                }
            } else {
                this.data = data.data;
                this.item = data.layout;
                this.form_items_joined();
            }

        });
    }

    ngOnDestroy(): void {
        if (this.interval) {
            clearInterval(this.interval);
        }
    }

}
