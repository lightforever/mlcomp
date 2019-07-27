import {Component, OnDestroy, OnInit} from '@angular/core';
import {AuxiliaryService} from "./auxiliary.service";
import {Auxiliary} from "../models";

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
            this.data = data;
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
