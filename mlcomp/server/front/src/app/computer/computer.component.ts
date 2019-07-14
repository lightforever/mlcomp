import {AfterViewInit, Component, OnChanges, OnInit, ViewChild} from '@angular/core';
import {Paginator} from "../paginator";
import {Computer, ComputerFilter} from "../models";
import {Location} from "@angular/common";
import {DynamicresourceService} from "../dynamicresource.service";
import {Helpers} from "../helpers";
import {ComputerService} from "./computer.service";

@Component({
    selector: 'app-computer',
    templateUrl: './computer.component.html',
    styleUrls: ['./computer.component.css']
})
export class ComputerComponent extends Paginator<Computer> implements AfterViewInit {

    displayed_columns: string[] = ['main', 'usage_history'];
    @ViewChild('table') table;
    last_time = {};
    plotted: boolean;
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
        this.plotted = false;
        this.pressed = event.value;
    };

    constructor(protected service: ComputerService, protected location: Location,
                protected resource_service: DynamicresourceService
    ) {
        super(service, location);
        this.id_column = 'name';
    }

    get_filter(): any {
        let res = new ComputerFilter();
        res.paginator = super.get_filter();
        res.usage_min_time = new Date(Date.now() - this.intervals[this.pressed] * 1000);
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
                let data = res.data;
                this.resource_service.load('plotly').then(() => {
                    setTimeout(() => {
                        let rendered = true;
                        for (let computer of data) {
                            let id = 'usage_history_' + computer.name;
                            if (!document.getElementById(id)) {
                                rendered = false;
                                break
                            }
                            let series = [];
                            for (let item of computer.usage_history.mean) {
                                let x = computer.usage_history.time.map(x => new Date(Date.parse(x)));
                                if (self.last_time[computer.name] && self.last_time[computer.name] >= x[x.length - 1]) {
                                    continue;
                                }
                                series.push({
                                    x: x,
                                    y: item.value,
                                    type: 'scatter',
                                    name: item.name
                                });

                            }

                            if (series.length > 0) {
                                self.last_time[computer.name] = new Date(Date.parse(series[0].x[series[0].x.length - 1]));
                            }

                            if (series.length > 0){
                                if (self.plotted) {
                                    let indices = Array.from(Array(series.length).keys());
                                    let y = {'y': [], 'x': []};
                                    for (let s of series) {
                                        y['y'].push(s.y);
                                        y['x'].push(s.x);
                                    }
                                    window['Plotly'].extendTraces(id, y, indices);
                                } else {
                                    window['Plotly'].newPlot(id, series, {}, {showSendToCloud: true});
                                    self.plotted = true;
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
        if(Helpers.parse_time(docker.last_activity)>=new Date(new Date()-10000)){
            return 'circle-green';
        }
        return 'circle-red';
    }

    long_date_format(time: string) {
        return Helpers.format_date_time(Helpers.parse_time(time));
    }
}
