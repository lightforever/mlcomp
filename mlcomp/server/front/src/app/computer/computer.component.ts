import {AfterViewInit, Component, OnChanges, OnInit, ViewChild} from '@angular/core';
import {Paginator} from "../paginator";
import {Computer} from "../models";
import {Location} from "@angular/common";
import {ComputerService} from "../computer.service";
import {DynamicresourceService} from "../dynamicresource.service";

@Component({
    selector: 'app-computer',
    templateUrl: './computer.component.html',
    styleUrls: ['./computer.component.css']
})
export class ComputerComponent extends Paginator<Computer> {

    displayed_columns: string[] = ['name', 'cpu', 'memory', 'gpu', 'usage_history'];
    @ViewChild('table') table;

    constructor(protected service: ComputerService, protected location: Location,
                protected resource_service: DynamicresourceService
    ) {
        super(service, location);
    }

    ngAfterViewInit() {
        this.data_updated.subscribe((data) => {
            this.resource_service.load('plotly').then(() => {
                setTimeout(() => {
                    while (true){
                        let rendered = true;
                        for (let computer of data) {
                            let id = 'usage_history_' + computer.name;
                            if(!document.getElementById(id)){
                                rendered = false;
                                break
                            }
                            let data = [];
                            for (let item of computer.usage_history.mean) {
                                data.push({
                                    x: computer.usage_history.time,
                                    y: item.value,
                                    type: 'scatter',
                                    name: item.name
                                })
                            }
                            window['Plotly'].newPlot(id, data, {}, {showSendToCloud: true});
                        }

                        if(rendered){
                            break;
                        }
                    }


                }, 100);


            });
        });
    }

    color(value: number) {
        if (value <= 33) {
            return 'green'
        }
        if (value <= 75) {
            return 'orange'
        }

        return 'red'
    }
}
