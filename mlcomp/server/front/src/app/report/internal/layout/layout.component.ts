import {Component, Input, OnInit} from '@angular/core';
import {Report, ReportItem} from "../../../models";
import {AppSettings} from "../../../app-settings";
import {ReportService} from "../../../report.service";

@Component({
    selector: 'app-layout',
    templateUrl: './layout.component.html',
    styleUrls: ['./layout.component.css']
})
export class LayoutComponent implements OnInit {

    @Input() item: ReportItem;
    @Input() data;
    @Input() report_id: number;
    @Input() id: string;
    items_joined: ReportItem[] = [];

    constructor(protected service: ReportService) {
    }

    ngOnInit() {
        this.subscribe_report_changed();
        this.form_items_joined();
    }

    form_items_joined(){
        this.items_joined = [];
        if (!this.item.items) {
            return;
        }
        for (let child of this.item.items) {
            if (child.type == 'series') {
                let subchildren = this.series(child);
                for (let s of subchildren) {
                    this.items_joined.push(s);
                }
            }
            else if(['img_classify', 'precision_recall', 'f1'].indexOf(child.type)!=-1){
                for(let d of this.data[child.source]){
                    this.items_joined.push(d);
                }
            }
            else {
                this.items_joined.push(child)
            }
        }
    }

    get item_data(){
        return this.item.data;
    }
    randid() {
        return 'series_' + Math.random().toString();
    }

    private series_single_task(data) {
        let plot = {'name': data[0].source +' - '+data[0].task_name, 'data': []};
        for (let d of data) {
            d.data.name = d.group;
            plot.data.push(d.data);
        }
        return [plot];
    }

    private series_multi_key(data, key) {
        let by_key = {};
        for (let d of data) {
            if (!(d[key] in by_key)) {
                by_key[d[key]] = {'name': d.source + ' - ' + (key == 'group' ? d[key] : d['task_name']), 'data': []};
            }
            d.data.name = key == 'task_id' ? d.group : d.task_name;
            by_key[d[key]].data.push(d.data);
        }

        let res = [];
        for (let k in by_key) {
            res.push(by_key[k]);
        }
        return res;
    }

    private unique(array, key) {
        let els = [];
        for (let item of array) {
            if (els.indexOf(item[key]) == -1) {
                els.push(item[key]);
            }
        }

        return els;
    }

    private series(item) {
        let data = this.data[item.source];
        let tasks = this.unique(data, 'task_id');
        if (tasks.length == 0) {
            return;
        }

        data = data.filter(d => !item.group || item.group.indexOf(d.group) != -1);

        let plots = [];
        if (tasks.length == 1) {
            plots = this.series_single_task(data);
        } else if (item.multi) {
            plots = this.series_multi_key(data, 'task_id');
        } else {
            plots = this.series_multi_key(data, 'group');
        }
        let res = [];
        for (let p of plots) {
            let id = this.randid();
            let resitem = new ReportItem();
            resitem.id = id;
            resitem.type = 'series';
            resitem.rows = item.rows;
            resitem.cols = item.cols;
            res.push(resitem);

            this.series_display(p, id);
        }

        return res;
    }

    private series_display(plot, id) {
        setTimeout(() => {
            if (document.getElementById(id)) {
                let layout = {
                    'title': plot.name,
                    'height': 300,
                    'width': 600,
                    'margin': {'b': 40, 't': 40}
                };
                for (let row of plot.data) {
                    let text = [];
                    for (let time of row.time) {
                        time = new Date(Date.parse(time));
                        text.push(AppSettings.format_date_time(time));
                    }

                    row.text = text;
                }
                window['Plotly'].newPlot(id, plot.data, layout);


            }
        }, 100);
    }

    td_width(child: ReportItem) {
        let total = 0;
        for (let c of this.items_joined) {
            total += c.cols ? c.cols : 1;
        }

        return (100 * (child.cols ? child.cols : 1) / total).toString() + '%';
    }

    private subscribe_report_changed() {
        if (this.report_id!=null) {
            this.service.data_updated.subscribe(res => {
                this.service.get_obj<Report>(this.report_id).subscribe(data => {
                    this.data = data.data;
                    this.item = data.layout;
                    this.form_items_joined();
                });
            });
        }
    }
}
