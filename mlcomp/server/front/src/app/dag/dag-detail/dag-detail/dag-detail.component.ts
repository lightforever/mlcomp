import {Component} from '@angular/core';
import {DagFilter} from "../../../models";
import {DagsComponent} from "../../dags/dags.component";

@Component({
    selector: 'app-dag-detail',
    templateUrl: './dag-detail.component.html',
    styleUrls: ['../../dags/dags.component.css']
})
export class DagDetailComponent extends DagsComponent {
    get_filter(): any {
        let res = super.get_filter();
        res.id = parseInt(this.route.snapshot.paramMap.get('id'));
        return res;
    }

    onActivate(component) {
        component.dag = parseInt(this.route.snapshot.paramMap.get('id'));
    }
}
