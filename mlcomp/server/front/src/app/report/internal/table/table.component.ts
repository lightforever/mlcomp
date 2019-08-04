import {Component, Input, OnInit} from '@angular/core';
import {Metric, ReportItem, Series} from "../../../models";
import {LayoutService} from "../layout/layout.service";

@Component({
    selector: 'app-table',
    templateUrl: './table.component.html',
    styleUrls: ['./table.component.css']
})
export class TableComponent implements OnInit {
    @Input() item: ReportItem;
    @Input() data;
    @Input() metric: Metric;

    rows: string[][];
    header: string[];

    sort_column: string;
    sort_desc: boolean;

    constructor(
        protected layout_service: LayoutService
    ) {
    }


    ngOnInit(): void {
        this.update();
        this.subscribe_data_changed();
    }

    _best_epoch(items) {
        let best_epoch = null;
        let best_score = this.metric.minimize ? 10 ** 6 : -(10 ** 6);
        for (let item of items) {
            if (item.group == 'valid' && item.name == this.metric.name) {
                for (let epoch = 0; epoch < item.y.length; epoch++) {
                    let is_best = false;
                    let v = item.y[epoch];
                    if (v > best_score && !this.metric.minimize) {
                        is_best = true;
                    } else if (v < best_score && this.metric.minimize) {
                        is_best = true;
                    }

                    if (is_best) {
                        best_score = v;
                        best_epoch = epoch;
                    }
                }
                break
            }
        }

        return best_epoch;
    }

    subscribe_data_changed() {
        this.layout_service.full_updated.subscribe(event => {
            this.update();
        });
    }

    update() {
        let data = [];
        for (let k in this.data) {
            if (this.item.source.indexOf(k) != -1 || k == this.metric.name) {
                for (let s of this.data[k]) {
                    data.push(s);
                }
            }
        }

        let task_group = data.reduce((g: any, s: Series) => {
            g[s.task_id] = g[s.task_id] || [];
            g[s.task_id].push(s);
            return g;
        }, {});

        let header = [];
        let rows = [];
        for (let task_id in task_group) {
            let items = task_group[task_id];
            let best_epoch = this._best_epoch(items);

            if (best_epoch == null) {
                continue;
            }

            let task_name = items[0].task_name;
            if (header.length == 0) {
                header.push('model');
                header.push('epoch');
            }


            let row = [task_name, best_epoch];
            for (let i = row.length; i < header.length; i++) {
                row.push(null);
            }

            let name_group = items.reduce(
                (g: any, s: Series) => {
                    g[s.name] = g[s.name] || [];
                    g[s.name].push(s);
                    return g;
                }, {});

            for (let name in name_group) {
                if (this.item.source.indexOf(name) == -1) {
                    continue;
                }

                let group = name_group[name].reduce(
                    (g: any, s: Series) => {
                        g[s.group] = g[s.group] || [];
                        g[s.group].push(s);
                        return g;
                    }, {});

                for (let group_name in group) {
                    let s = group[group_name][0];
                    let h = name + '_' + group_name;
                    let idx = header.indexOf(h);
                    if (idx == -1) {
                        header.push(h);
                        row.push(s.y[best_epoch]);
                    } else {
                        row[idx] = s.y[best_epoch];
                    }

                }
            }

            rows.push(row);
        }

        this.header = header;
        this.rows = rows;
        if (this.sort_desc == null) {
            this.sort_desc = this.metric.minimize;
        }

        this.sort();
    }

    sort() {
        let self = this;
        let column = this.sort_column ?
            this.sort_column :
            this.metric.name + '_' + 'valid';

        let key_idx = this.header.indexOf(column);

        function sort_row(a, b) {
            if (self.sort_desc) {
                return a[key_idx] < b[key_idx] ? -1 : 1;
            }
            return a[key_idx] > b[key_idx] ? -1 : 1;
        }

        this.rows.sort(sort_row);
    }

    is_number(n) {
        return !isNaN(parseFloat(n)) && isFinite(n);
    }
}