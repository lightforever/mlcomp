import {Component} from '@angular/core';
import {Paginator} from "../paginator";
import {Model, ModelFilter} from "../models";
import {Location} from "@angular/common";
import {ActivatedRoute, Router} from "@angular/router";
import {ModelService} from "../model.service";

@Component({
    selector: 'app-model',
    templateUrl: './model.component.html',
    styleUrls: ['./model.component.css']
})
export class ModelComponent extends Paginator<Model> {
    protected displayed_columns: string[] = ['name', 'score_local', 'score_public'];
    name: string;
    project: number;

    constructor(protected service: ModelService, protected location: Location,
                protected router: Router, protected  route: ActivatedRoute
    ) {
        super(service, location);
    }

    filter_name(name: string) {
        this.name = name;
        this.change.emit();
    }

    get_filter(): any {
        let res = new ModelFilter();
        res.paginator = super.get_filter();
        res.paginator.sort_column = 'created';

        res.name = this.name;
        res.project = this.project;
        return res;
    }


}
