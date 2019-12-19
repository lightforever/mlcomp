import {AfterViewInit, Component, ViewChild} from '@angular/core';
import {Paginator} from "../paginator";
import {Computer, ComputerFilter} from "../models";
import {Location} from "@angular/common";
import {DynamicresourceService} from "../dynamicresource.service";
import {Helpers} from "../helpers";
import {ComputerService} from "./computer.service";
import {SyncDialogComponent} from "./sync-dialog";
import {MatDialog} from "@angular/material/dialog";

@Component({
    selector: 'app-computer',
    templateUrl: './computer.component.html',
    styleUrls: ['./computer.component.css']
})
export class ComputerComponent extends Paginator<Computer>
    implements AfterViewInit {

    displayed_columns: string[] = [
        'main',
        'usage_history'
    ];

    @ViewChild('table') table;
    last_time = {};
    public pressed: string = 'min15';
    intervals = {
        min15: 15 * 60,
        hour1: 60 * 60,
        hour3: 60 * 60 * 3,
        hour6: 60 * 60 * 6,
        day1: 60 * 60 * 24,
        day3: 60 * 60 * 24 * 3,
    };

    pressed_changed(event) {
        this.last_time = {};
        this.pressed = event.value;
        for (let c of this.dataSource.data) {
            let id = 'usage_history_' + c.name;
            window['Plotly'].purge(id);
        }
        this.change.emit();
    };

    constructor(protected service: ComputerService,
                protected location: Location,
                protected resource_service: DynamicresourceService,
                public sync_dialog: MatDialog,
    ) {
        super(service, location);
        this.id_column = 'name';
    }

    get_filter(): any {
        let res = new ComputerFilter();
        res.paginator = super.get_filter();
        res.usage_min_time = new Date(Date.now() -
            this.intervals[this.pressed] * 1000);

        for (let key in this.last_time) {
            if (this.last_time[key] > res.usage_min_time) {
                res.usage_min_time = this.last_time[key];
            }
        }
        return res;
    }

    ngAfterViewInit() {
        let self = this;
        this.data_updated.subscribe((res) => {
                if (!res || !res.data) {
                    return;
                }
                let data = res.data;
                this.resource_service.load('plotly').then(() => {
                    setTimeout(() => {
                        let rendered = true;
                        for (let computer of data) {
                            let id = 'usage_history_' + computer.name;
                            let element = document.getElementById(id);
                            if (!element) {
                                rendered = false;
                                break
                            }
                            let series = [];
                            for (let item of computer.usage_history.mean) {
                                let x = computer.usage_history.time.map(x => new Date(Date.parse(x)));


                                let last = self.last_time[computer.name];
                                if (last && last >= x[x.length - 1]) {
                                    continue;
                                }
                                series.push({
                                    x: x,
                                    y: item.value,
                                    type: 'scatter',
                                    name: item.name,
                                    visible: item.name == 'disk' ?
                                        'legendonly' : true
                                });

                            }

                            if (series.length > 0) {
                                let x1 = series[0].x[series[0].x.length - 1];
                                let parsed = Date.parse(x1);
                                self.last_time[computer.name] = new Date(
                                    parsed
                                );
                            }

                            if (series.length > 0) {
                                if (element.childNodes.length > 0) {
                                    let keys = Array(series.length).keys();
                                    let indices = Array.from(keys);
                                    let y = {'y': [], 'x': []};
                                    for (let s of series) {
                                        y['y'].push(s.y);
                                        y['x'].push(s.x);
                                    }
                                    window['Plotly'].extendTraces(id,
                                        y,
                                        indices);

                                } else {
                                    window['Plotly'].newPlot(id, series, {},
                                        {showSendToCloud: true});
                                }
                            }


                        }

                    }, 100);


                });
            }
        );
    }

    color(value
              :
              number
    ) {
        if (value <= 33) {
            return 'green'
        }
        if (value <= 75) {
            return 'orange'
        }

        return 'red'
    }

    docker_status(docker) {
        // @ts-ignore
        if (Helpers.parse_time(docker.last_activity) >= new Date(
            Date.now() - 15000)) {
            return 'circle-green';
        }
        return 'circle-red';
    }

    docker_status_tip(docker) {
        if (Helpers.parse_time(docker.last_activity) >= new Date(
            Date.now() - 15000)) {
            return 'online';
        }

        return 'offline';
    }

    long_date_format(time: string) {
        return Helpers.format_date_time(Helpers.parse_time(time));
    }

    sync(name: string) {
        this.service.sync_start().subscribe(x => {
            let dialog = this.sync_dialog.open(SyncDialogComponent,
                {
                    width: '500px', height: '500px',
                    data: x
                });
            dialog.afterClosed().subscribe(res => {
                res['computer'] = name;
                this.service.sync_end(res).subscribe(x=>{});
            });
        });
    }
}
