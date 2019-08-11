import {Component, Input, OnInit} from '@angular/core';
import {ReportItem, SeriesItem, Series} from "../../../models";
import {Helpers} from "../../../helpers";
import {LayoutService} from "../layout/layout.service";

@Component({
    selector: 'app-series',
    templateUrl: './series.component.html',
    styleUrls: ['./series.component.css']
})
export class SeriesComponent implements OnInit {

    @Input() item: ReportItem;
    @Input() data: SeriesItem;
    private id = 'series_' + Math.random().toString();

    constructor(
        protected layout_service: LayoutService
    ) {
    }

    ngOnInit() {
        this.display();
        this.subscribe_data_changed();
    }

    private display() {
        setTimeout(() => {
            if (document.getElementById(this.id)) {
                let layout = {
                    'title': this.data.name,
                    'height': 300,
                    'width': 600,
                    'margin': {'b': 40, 't': 40},
                    'shapes': [],
                    'annotations': []
                };
                for (let row_idx = 0; row_idx < this.data.series.length;
                     row_idx++) {
                    let row = this.data.series[row_idx];
                    let text = [];
                    let last_stage = '';
                    for (let i = 0; i < row.time.length; i++) {
                        let t = new Date(Date.parse(row.time[i]));
                        text.push(Helpers.format_date_time(t));

                        if (row_idx == 0 &&
                            last_stage &&
                            last_stage != row.stage[i]) {
                            layout.shapes.push({
                                type: 'line',
                                color: 'green',
                                yref: 'paper',
                                x0: row.x[i] - 1,
                                y0: 0,
                                x1: row.x[i] - 1,
                                y1: 1,
                                text: 'text',
                                line: {
                                    color: 'green',
                                    width: 1.5
                                }
                            });

                            layout.annotations.push(
                                {
                                    x: row.x[i] - 1,
                                    y: 0,
                                    showarrow: false,
                                    yref: "paper",
                                    text: row.stage[i],
                                })
                        }
                        last_stage = row.stage[i];
                    }

                    row.text = text;
                }

                if(this.data.layout && JSON.stringify(this.data.layout) !=
                    JSON.stringify(layout)){
                    for(let s of this.data.series){
                        s.plotted = 0;
                    }
                }
                this.data.layout = layout;

                if (this.data.series.length > 0) {
                    if (this.data.series.map(
                        x => x.plotted).reduce((s, c) => s + c, 0) > 0) {
                        let keys = Array(this.data.series.length).keys();
                        let indices = Array.from(keys);
                        let y = {'y': [], 'text': []};
                        for (let s_idx = 0; s_idx < this.data.series.length;
                             s_idx++
                        ) {
                            let s = this.data.series[s_idx];
                            y['y'].push(s.y.slice(s.plotted));
                            let text = s.text.slice(s.plotted);
                            y['text'].push(text);

                            s.plotted += text.length;

                        }
                        window['Plotly'].extendTraces(this.id, y, indices);
                    } else {
                        window['Plotly'].newPlot(this.id,
                            this.data.series,
                            layout);

                        for (let s of this.data.series) {
                            s.plotted = s.x.length;
                        }
                    }
                }


            }
        }, 100);
    }

    private subscribe_data_changed() {
        this.layout_service.data_updated.subscribe(event => {
            if (event.key != this.item.source) {
                return;
            }

            let was_change = false;
            for (let i = 0; i < this.data.series.length; i++) {
                let d = this.data.series[i];
                for (let series of event.data) {
                    if (series.task_id == d.task_id &&
                        series.group == d.group && series.source == d.source) {
                        if (series.x.length > this.data.series[i].x.length) {
                            this.data.series[i].x = series.x;
                            this.data.series[i].y = series.y;
                            this.data.series[i].time = series.time;
                            this.data.series[i].stage = series.stage;

                            was_change = true;
                            break
                        }

                    }
                }
            }

            if (was_change) {
                this.display();
            }

        });
    }


    private static create_single_task(data) {
        let first = data[0];
        let plot: SeriesItem = {
            'name': first.source +
                ' - ' + first.task_name,
            'series': [],
            'layout': null
        };

        for (let d of data) {
            d = Helpers.clone(d);
            d.plotted = 0;
            d.name = d.group;
            plot.series.push(d);
        }
        return [plot];
    }

    private static create_multi_key(data, key) {
        let by_key = {};
        for (let d of data) {
            d = Helpers.clone(d);
            if (!(d[key] in by_key)) {
                by_key[d[key]] = {
                    'name': d.source + ' - ' +
                        (key == 'group' ?
                            d[key] :
                            d['task_name']), 'series': []
                };
            }
            d.name = key == 'task_id' ? d.group : d.task_name;
            d.plotted = 0;
            by_key[d[key]].series.push(d);
        }

        let res: SeriesItem[] = [];
        for (let k in by_key) {
            res.push(by_key[k]);
        }
        return res;
    }

    public static create(item: ReportItem, data: Series[]) {
        if (!data || data.length == 0) {
            return []
        }
        let tasks = Helpers.unique(data, 'task_id');
        if (tasks.length == 0) {
            return [];
        }

        data = data.filter(d => !item.group ||
            item.part.indexOf(d.group) != -1);

        let plots: SeriesItem[] = [];
        if (tasks.length == 1) {
            plots = this.create_single_task(data);
        } else if (item.multi) {
            plots = this.create_multi_key(data, 'task_id');
        } else {
            plots = this.create_multi_key(data, 'group');
        }
        let items = [];
        for (let p of plots) {
            let resitem = Helpers.clone(item);
            items.push([resitem, p]);
        }

        return items;
    }

}
