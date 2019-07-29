import {Component, OnDestroy, OnInit} from '@angular/core';
import {AuxiliaryService} from "./auxiliary.service";
import {Auxiliary} from "../models";
import {Helpers} from "../helpers";

@Component({
    selector: 'app-auxiliary',
    templateUrl: './auxiliary.component.html',
    styleUrls: ['./auxiliary.component.css']
})
export class AuxiliaryComponent implements OnInit, OnDestroy {
    data: Auxiliary;
    private interval: number;

    constructor(
        protected service: AuxiliaryService
    ) {
    }

    load() {
        this.service.get_obj<Auxiliary>({}).subscribe(data => {
            this.data = Helpers.update_object(this.data,
                data,
                [
                    'supervisor.computers.name',
                    'supervisor.parent_tasks_stats.id'
                ]);
        })
    }

    ngOnInit() {
        this.load();

        this.interval = setInterval(
            () => this.load(),
            3000);

    }

    ngOnDestroy() {
        clearInterval(this.interval);

    }

}
