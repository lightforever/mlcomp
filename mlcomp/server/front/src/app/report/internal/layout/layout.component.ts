import {Component, Input, OnDestroy, OnInit} from '@angular/core';
import {Metric, Report, ReportItem} from "../../../models";
import {LayoutService} from "./layout.service";
import {SeriesComponent} from "../series/series.component";
import {Helpers} from "../../../helpers";
import {ReportService} from "../../report.service";

@Component({
    selector: 'app-layout',
    templateUrl: './layout.component.html',
    styleUrls: ['./layout.component.css']
})
export class LayoutComponent implements OnInit, OnDestroy {
    _item: ReportItem;
    _data;

    @Input() report_id: number;
    @Input() id: string;
    @Input() metric: Metric;

    private interval: number;
    items_joined: ReportItem[] = [];
    items_joined_data: any[] = [];

    constructor(protected service: ReportService,
                protected layout_service: LayoutService) {
    }

    ngOnInit() {
        this.form_items_joined();
        this.subscribe_report_changed();
    }

    @Input()
    set item(item: ReportItem) {
        this._item = item;
        this.form_items_joined();
    }

    get item() {
        return this._item;
    }

    @Input()
    set data(data) {
        this._data = data;
    }

    get data() {
        return this._data;
    }

    form_items_joined() {
        this.items_joined = [];
        this.items_joined_data = [];

        if (!this.item || !this.item.items) {
            return;
        }
        if (!this.data) {
            return;
        }
        for (let child of this.item.items) {
            if (child.type == 'series') {
                if (child.source == '_other') {
                    continue;
                }
                let subchildren = SeriesComponent.create(child,
                    this.data[child.source]);

                for (let s of subchildren) {
                    this.items_joined.push(s[0]);
                    this.items_joined_data.push(s[1]);

                    this.data[child.source].mapped = true;
                }
            } else {
                let single_elements = ['img_classify', 'img', 'img_segment'];
                if (single_elements.indexOf(child.type) != -1) {
                    if (!(child.source in this.data)) {
                        continue
                    }
                    let i = 0;
                    for (let d of this.data[child.source]) {
                        let child_clone = Helpers.clone(child);
                        child_clone.index = i;
                        this.items_joined.push(child_clone);
                        this.items_joined_data.push(d);
                        i++;
                    }
                } else if (child.type == 'table') {
                    this.items_joined.push(child);
                    this.items_joined_data.push(this.data);
                } else {
                    this.items_joined.push(child);
                    this.items_joined_data.push(this.data);
                }
            }
        }

        setTimeout(() => {
            for (let child of this.item.items) {
                if (child.type == 'series' && child.source == '_other') {
                    for (let name of Object.getOwnPropertyNames(this.data)) {
                        let value = this.data[name];
                        if (value.mapped) {
                            continue
                        }
                        if (!Array.isArray(value) || value.length == 0 || !value[0].source) {
                            continue
                        }
                        let subchildren = SeriesComponent.create(child, value);

                        for (let s of subchildren) {
                            s[0].source = name;

                            this.items_joined.push(s[0]);
                            this.items_joined_data.push(s[1]);
                        }
                    }

                }
            }
        }, 500);
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
            this.interval = setInterval(() =>
                this.update(), 5000);

            this.service.data_updated.subscribe(res => {
                this.update(true);
            });
        }
    }

    private update(hard = false) {
        this.service.get_obj<Report>(this.report_id).subscribe(
            data => {
                if (this.item && !hard) {
                    if (JSON.stringify(this.item) !=
                        JSON.stringify(data.layout)
                    ) {
                        this._data = data.data;
                        this._item = data.layout;
                        this.form_items_joined();
                    } else {
                        for (let key in data.data) {
                            // noinspection JSUnfilteredForInLoop
                            if (JSON.stringify(this.data[key]) !=
                                JSON.stringify(data.data[key])) {
                                if (data.data[key].length !=
                                    this.data[key].length
                                ) {
                                    this._data = data.data;
                                    this._item = data.layout;
                                    this.form_items_joined();
                                    break;
                                } else {
                                    // noinspection JSUnfilteredForInLoop
                                    let value = {
                                        'key': key,
                                        'data': data.data[key]
                                    };
                                    this.layout_service.data_updated.emit(
                                        value
                                    );
                                    // noinspection JSUnfilteredForInLoop
                                    this.data[key] = data.data[key];
                                }

                            }
                        }
                    }
                } else {
                    this._data = data.data;
                    this._item = data.layout;
                    this.form_items_joined();
                }

                this.layout_service.full_updated.emit(this.data);
            });
    }

    ngOnDestroy(): void {
        if (this.interval) {
            clearInterval(this.interval);
        }
    }

}
